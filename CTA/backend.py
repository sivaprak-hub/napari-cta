import os
import struct
import numpy as np
import pandas as pd
import tifffile
import traceback
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, butter, sosfiltfilt, hilbert
from scipy.interpolate import interp1d
from skimage.measure import block_reduce
from skimage.filters import threshold_otsu
from sklearn.preprocessing import StandardScaler
from qtpy.QtCore import QThread, Signal
import warnings

try:
    from aicsimageio import AICSImage
except Exception:
    AICSImage = None

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Olympus ETS reader
# ---------------------------------------------------------------------------

def _find_ets_files(vsi_path):
    """Return sorted list of .ets files in the VSI companion folder."""
    base = os.path.splitext(os.path.basename(vsi_path))[0]
    companion = os.path.join(os.path.dirname(vsi_path), f'_{base}_')
    if not os.path.isdir(companion):
        return []
    results = []
    for root, _dirs, files in os.walk(companion):
        for f in files:
            if f.lower().endswith('.ets'):
                results.append(os.path.join(root, f))
    return sorted(results)


def _read_ets(ets_path):
    """
    Read all frames from an Olympus ETS (Encoded Tile Sequence) file.

    Binary layout:
      0-63   SIS outer header  — dir_offset at [32], dir_count at [40]
      64-127 ETS sub-header    — pixeltype at [72], width at [92], height at [96]
      ...    frame data (sequential 1-MB blocks from ~offset 292)
      EOF-N  tile directory    — dir_count × 44-byte entries

    Each 44-byte entry: struct '<11I'
      (type, pad, pad, T_idx, pad, pad, pad, offset_lo, offset_hi, size, seq)

    Returns (T, H, W) numpy array.
    """
    file_size = os.path.getsize(ets_path)
    with open(ets_path, 'rb') as fh:
        hdr = fh.read(128)

    if hdr[:3] != b'SIS':
        raise ValueError(f"Not a valid ETS file (bad magic): {ets_path}")

    dir_offset = struct.unpack_from('<Q', hdr, 32)[0]
    dir_count  = struct.unpack_from('<Q', hdr, 40)[0]
    pixeltype  = struct.unpack_from('<I', hdr, 72)[0]
    width      = struct.unpack_from('<I', hdr, 92)[0]
    height     = struct.unpack_from('<I', hdr, 96)[0]

    with open(ets_path, 'rb') as fh:
        fh.seek(dir_offset)
        raw_dir = fh.read(file_size - dir_offset)

    entries = []
    for i in range(int(dir_count)):
        e      = struct.unpack_from('<11I', raw_dir, i * 44)
        t_idx  = e[3]
        offset = e[7] | (e[8] << 32)
        size   = e[9]
        entries.append((t_idx, offset, size))

    if not entries:
        raise ValueError(f"ETS file contains no tile entries: {ets_path}")

    # Derive dtype from frame byte size (avoid relying on pixeltype field)
    frame_pixels = width * height
    sample_size  = entries[0][2]
    if sample_size == frame_pixels:
        dtype = np.uint8
    elif sample_size == frame_pixels * 2:
        dtype = np.uint16
    else:
        dtype = np.uint8

    entries_sorted = sorted(entries, key=lambda e: e[0])
    n_frames = len(entries_sorted)

    result = np.empty((n_frames, height, width), dtype=dtype)
    with open(ets_path, 'rb') as fh:
        for i, (_t, offset, size) in enumerate(entries_sorted):
            fh.seek(offset)
            raw = fh.read(size)
            result[i] = np.frombuffer(raw, dtype=dtype).reshape(height, width)

    return result


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def load_image(path):
    if path.lower().endswith('.vsi'):
        ets_files = _find_ets_files(path)
        if ets_files:
            # Use the largest ETS file (main time-series stack)
            ets_path = max(ets_files, key=os.path.getsize)
            return _read_ets(ets_path)
        # Fallback: read the TIFF thumbnail via aicsimageio
        if AICSImage is None:
            raise ImportError("aicsimageio not installed. Run: pip install aicsimageio")
        img = AICSImage(path)
        return img.get_image_data("TYX")

    with tifffile.TiffFile(path) as tif:
        data = tif.asarray()
        # Try to read axes order from metadata (ImageJ / OME-TIFF)
        axes = ''
        try:
            axes = tif.series[0].axes.upper()
        except Exception:
            pass

    if data.ndim == 2:
        return data[np.newaxis, ...]          # (Y,X) → (1,Y,X)

    if data.ndim == 3:
        return data                           # assume (T,Y,X)

    if data.ndim == 4:
        # Determine which axis is channels/z vs time using axes metadata or heuristics
        if axes and 'C' in axes:
            c = axes.index('C')
            data = np.take(data, 0, axis=c)  # drop channel → (T,Y,X)
        elif axes and 'T' in axes and 'Z' in axes:
            z = axes.index('Z')
            data = np.take(data, 0, axis=z)  # drop Z → (T,Y,X)
        elif axes and len(axes) == 4:
            # Use axes order to extract TYX slice
            try:
                t_pos = axes.index('T')
                y_pos = axes.index('Y')
                x_pos = axes.index('X')
                other = [i for i in range(4) if i not in (t_pos, y_pos, x_pos)][0]
                data = np.take(data, 0, axis=other)
            except Exception:
                data = data[0]
        else:
            # Heuristic: smallest dimension is likely channels
            small_ax = int(np.argmin(data.shape))
            if data.shape[small_ax] <= 4:
                data = np.take(data, 0, axis=small_ax)
            else:
                data = data[0]
        return data if data.ndim == 3 else data[0]

    raise ValueError(
        f"Unsupported TIFF shape {data.shape}. Expected (T,Y,X), (T,C,Y,X), or similar."
    )


def read_file_timing(path):
    """
    Extract FPS from file metadata.
    For VSI: returns frame count T from ETS companion file; FPS must be set manually.
    For TIFF: checks ImageJ, OME-TIFF, MicroManager, and Olympus cellSens metadata.
    Returns dict with keys: fps (float or None), T (int or None), source (str or None).
    """
    result = {'fps': None, 'T': None, 'source': None}

    if path.lower().endswith('.vsi'):
        try:
            ets_files = _find_ets_files(path)
            if ets_files:
                ets_path = max(ets_files, key=os.path.getsize)
                with open(ets_path, 'rb') as fh:
                    hdr = fh.read(48)
                result['T'] = int(struct.unpack_from('<Q', hdr, 40)[0])
        except Exception:
            pass
        return result

    if not path.lower().endswith(('.tif', '.tiff')):
        return result

    try:
        with tifffile.TiffFile(path) as tif:
            # Frame count from series
            try:
                result['T'] = int(tif.series[0].shape[0])
            except Exception:
                pass

            # --- 1. ImageJ metadata ---
            ij = tif.imagej_metadata or {}
            fi = ij.get('finterval')
            if fi and float(fi) > 0:
                result['fps']    = round(1.0 / float(fi), 4)
                result['source'] = 'ImageJ metadata (finterval)'
                return result
            fp = ij.get('fps')
            if fp and float(fp) > 0:
                result['fps']    = round(float(fp), 4)
                result['source'] = 'ImageJ metadata (fps)'
                return result

            # --- 2. OME-TIFF ---
            if tif.is_ome:
                try:
                    import xml.etree.ElementTree as ET
                    ome_xml = tif.ome_metadata
                    root = ET.fromstring(ome_xml)
                    # namespace-agnostic search
                    for pixels in root.iter():
                        if pixels.tag.endswith('Pixels'):
                            ti = pixels.get('TimeIncrement')
                            tu = pixels.get('TimeIncrementUnit', 's')
                            if ti:
                                ti = float(ti)
                                if tu in ('ms', 'millisecond', 'Milliseconds'):
                                    ti /= 1000.0
                                elif tu in ('min', 'Minutes'):
                                    ti *= 60.0
                                if ti > 0:
                                    result['fps']    = round(1.0 / ti, 4)
                                    result['source'] = 'OME-TIFF (TimeIncrement)'
                                    return result
                except Exception:
                    pass

            # --- 3. MicroManager ---
            mm = getattr(tif, 'micromanager_metadata', None)
            if mm:
                try:
                    interval_ms = mm.get('Summary', {}).get('Interval_ms')
                    if interval_ms and float(interval_ms) > 0:
                        result['fps']    = round(1000.0 / float(interval_ms), 4)
                        result['source'] = 'MicroManager (Interval_ms)'
                        return result
                except Exception:
                    pass

            # --- 4. Olympus cellSens (frame timestamps in ImageJ Info string) ---
            info_str = ij.get('Info', '')
            if info_str:
                try:
                    import re
                    # Parse all Value/Units pairs from the Info string
                    # Anchor to line-start to avoid matching 'Z valueValue/Units #N' lines
                    vals  = {int(m.group(1)): float(m.group(2))
                             for m in re.finditer(r'^Value #(\d+)\s*=\s*([\d.eE+\-]+)',
                                                  info_str, re.MULTILINE)}
                    units = {int(m.group(1)): m.group(2).strip()
                             for m in re.finditer(r'^Units #(\d+)\s*=\s*([^\n]+)',
                                                  info_str, re.MULTILINE)}
                    # Timestamps are Values where the matching Unit is ms (10^-3s^1)
                    ts_ms = [vals[i] for i in sorted(vals)
                             if '10^-3s^1' in units.get(i, '')]
                    if len(ts_ms) >= 2:
                        ts = np.array(ts_ms) / 1000.0   # → seconds
                        avg_interval = (ts[-1] - ts[0]) / (len(ts) - 1)
                        if avg_interval > 0:
                            result['fps']        = round(1.0 / avg_interval, 4)
                            result['source']     = 'Olympus cellSens (frame timestamps)'
                            result['timestamps'] = ts  # actual non-uniform timestamps
                            return result
                except Exception:
                    pass
    except Exception:
        pass

    return result


def convert_single_vsi(input_path):
    """
    Converts VSI to ImageJ TIFF. Reads frames from the ETS companion file when present.
    FPS metadata is not available in Olympus VSI; defaults to finterval=1.0 s.
    """
    try:
        ets_files = _find_ets_files(input_path)
        if ets_files:
            ets_path = max(ets_files, key=os.path.getsize)
            data = _read_ets(ets_path)
        elif AICSImage is not None:
            img = AICSImage(input_path)
            reader_dims = img.dims.order
            if "C" in reader_dims:
                data = img.get_image_data("TCYX")
                data = data[:, 0, :, :]
            else:
                data = img.get_image_data("TYX")
        else:
            return False, "No ETS companion file found and aicsimageio is not installed."

        save_path = os.path.splitext(input_path)[0] + ".tif"
        tifffile.imwrite(
            save_path,
            data,
            photometric='minisblack',
            metadata={'axes': 'TYX', 'finterval': 1.0},
            imagej=True,
        )
        return True, f"Saved {data.shape[0]} frames to {os.path.basename(save_path)}"

    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Photobleaching helpers
# ---------------------------------------------------------------------------

def _bleach_envelope(sig, n_pts):
    """
    Rolling 10th-percentile envelope — tracks slow photobleaching drift
    without being pulled up by transient peaks.
    """
    n   = len(sig)
    win = max(n // 8, 3)
    return np.array([
        np.percentile(sig[max(0, i - win // 2): i + win // 2 + 1], 10)
        for i in range(n)
    ])


def _single_exp(t, a, tau, c):
    tau = max(tau, 1e-5)
    return a * np.exp(-np.clip(t / tau, 0, 700)) + c


# ---------------------------------------------------------------------------
# Per-transient kinetics (unchanged from fixed version)
# ---------------------------------------------------------------------------

def get_time_at_level(time, signal, start_idx, end_idx, level, mode='rising'):
    """
    Returns interpolated time when signal crosses `level`.
    Returns np.nan when no crossing exists rather than silently returning t[0].
    """
    if start_idx >= end_idx:
        return np.nan
    segment  = signal[start_idx:end_idx]
    t_segment = time[start_idx:end_idx]
    try:
        if mode == 'rising':
            matches = np.where(segment >= level)[0]
        else:
            matches = np.where(segment <= level)[0]

        if len(matches) == 0:
            return np.nan

        i = matches[0]
        if i == 0:
            return t_segment[0]

        y1, y2 = segment[i - 1], segment[i]
        t1, t2 = t_segment[i - 1], t_segment[i]
        if y2 == y1:
            return t1
        return t1 + (level - y1) / (y2 - y1) * (t2 - t1)
    except Exception:
        return np.nan


def extract_detailed_features(time_stamps, signal):
    sig_range = np.max(signal) - np.min(signal)
    if sig_range < 1e-6:
        return None

    max_peaks, props = find_peaks(signal, prominence=sig_range * 0.15)
    if len(max_peaks) == 0:
        return None
    peak_idx = max_peaks[np.argmax(props['prominences'])]

    min_peaks, _ = find_peaks(-signal)
    pre      = min_peaks[min_peaks < peak_idx]
    start_idx = pre[-1] if len(pre) > 0 else 0
    post     = min_peaks[min_peaks > peak_idx]
    end_idx  = post[0] if len(post) > 0 else len(signal)

    baseline  = signal[start_idx]
    amp       = signal[peak_idx] - baseline
    t_start   = time_stamps[start_idx]
    peak_time = time_stamps[peak_idx]

    levs = [baseline + x * amp for x in [0.1, 0.5, 0.9]]

    # peak_idx + 1 so the peak sample is included in the rising window
    t_on  = [get_time_at_level(time_stamps, signal, start_idx, peak_idx + 1, l, 'rising')  for l in levs]
    t_off = [get_time_at_level(time_stamps, signal, peak_idx, end_idx,       l, 'decay')   for l in reversed(levs)]

    def dms(t1, t2):
        return abs(t1 - t2) * 1000 if not np.isnan(t1) and not np.isnan(t2) else np.nan

    cd = dms(t_off[2], t_on[0])
    cd_estimated = False
    if np.isnan(cd):
        w50 = dms(t_off[1], t_on[1])
        if not np.isnan(w50):
            cd = w50 * 1.6
            cd_estimated = True

    t_end = time_stamps[end_idx] if end_idx < len(time_stamps) else np.nan

    return {
        'BPM':      (len(max_peaks) / (time_stamps[-1] - time_stamps[0])) * 60
                    if (time_stamps[-1] - time_stamps[0]) > 0 else 0,
        'Amp':      amp,
        'F0':       baseline,
        'T_ON_ms':  (peak_time - t_start) * 1000,
        'T10_ON':   dms(t_on[0],  t_start),
        'T50_ON':   dms(t_on[1],  t_start),
        'T90_ON':   dms(t_on[2],  t_start),
        'T10_OFF':  dms(t_off[0], peak_time),
        'T50_OFF':  dms(t_off[1], peak_time),
        'T90_OFF':  dms(t_off[2], peak_time),
        'CD':       cd,
        'CD_estimated': cd_estimated,
        'T_OFF_ms': dms(t_end, peak_time),
    }


def calculate_synchronicity(signals):
    if signals.shape[0] < 2:
        return 0.0
    mean = np.mean(signals, axis=1, keepdims=True)
    std  = np.std(signals,  axis=1, keepdims=True)
    std[std == 0] = 1
    normalized = (signals - mean) / std
    corr = np.corrcoef(normalized)
    n    = corr.shape[0]
    off_diag = corr[np.triu_indices(n, k=1)]
    return float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0


# ---------------------------------------------------------------------------
# Spatiotemporal analysis pipeline
# ---------------------------------------------------------------------------

def compute_pulsatility_map(corrected_signals, H_bin, W_bin, T_bin, fps_eff):
    """
    Bandpass-filter each bin trace (0.3 – 5 Hz, clamped below Nyquist),
    then compute the variance of the filtered signal as the pulsatility score.

    Falls back to raw-signal variance when fps is too low for bandpass.

    Returns
    -------
    pulsatility_map : (H_bin, W_bin)  — high where cells are beating
    filtered_traces : (N, T_bin)      — bandpassed traces (or raw if filter skipped)
    """
    f_low  = 0.3
    f_high = 5.0
    nyq    = fps_eff / 2.0
    fh     = min(f_high, nyq * 0.9)

    filtered_traces = corrected_signals.copy()   # fallback: use raw corrected signal
    filter_applied  = False

    if f_low < fh and nyq > f_low:
        try:
            sos = butter(4, [f_low / nyq, fh / nyq], btype='band', output='sos')
            tmp = np.zeros_like(corrected_signals)
            n_ok = 0
            for i in range(len(corrected_signals)):
                try:
                    tmp[i] = sosfiltfilt(sos, corrected_signals[i])
                    n_ok += 1
                except Exception:
                    tmp[i] = corrected_signals[i]   # keep raw for this bin
            filtered_traces = tmp
            filter_applied  = n_ok > 0
        except Exception:
            pass   # filtered_traces already set to corrected_signals copy

    pulsatility_map = np.var(filtered_traces, axis=1).reshape(H_bin, W_bin)
    return pulsatility_map, filtered_traces


def compute_activity_mask(pulsatility_map):
    """
    Otsu threshold on the pulsatility map.
    The distribution is bimodal (dead/background vs active cells), so Otsu
    finds the natural separation automatically.

    Falls back to top-35% threshold if Otsu gives a degenerate result
    (< 3% or > 85% of bins marked active).
    """
    p_flat = pulsatility_map.ravel()

    # Degenerate: map is constant or all-zero → force top-35%
    if np.ptp(p_flat) < 1e-10:
        thresh = np.percentile(p_flat, 65)
        return pulsatility_map > thresh

    try:
        thresh = threshold_otsu(pulsatility_map)
        mask   = pulsatility_map > thresh
        frac   = np.mean(mask)
        if frac < 0.03 or frac > 0.85:
            # Otsu gave unreasonable result — use top-35%
            thresh = np.percentile(p_flat, 65)
            mask   = pulsatility_map > thresh
    except Exception:
        thresh = np.percentile(p_flat, 65)
        mask   = pulsatility_map > thresh

    return mask


def compute_reference_trace(corrected_signals, activity_mask):
    """Mean trace across active bins — used for global beat detection."""
    mask_flat = activity_mask.flatten()
    active    = corrected_signals[mask_flat]
    if len(active) == 0:
        return np.mean(corrected_signals, axis=0)
    return np.mean(active, axis=0)


def detect_beats(reference_trace, time_stamps, fps_eff):
    """
    Detect beat peaks in the reference trace.
    Minimum spacing = 0.8 s (handles up to ~75 BPM).
    Returns array of peak frame indices.
    """
    sig_range = np.max(reference_trace) - np.min(reference_trace)
    if sig_range < 1e-6:
        return np.array([], dtype=int)

    min_dist = max(int(fps_eff * 0.8), 2)
    peaks, _ = find_peaks(
        reference_trace,
        prominence=sig_range * 0.15,
        distance=min_dist,
    )
    return peaks


def compute_activation_time_map(corrected_signals, activity_mask, beat_peaks,
                                 time_stamps, fps_eff):
    """
    For each detected beat, find the frame at which each active bin first crosses
    50% of its peak amplitude (50%-rise activation time).

    All per-beat maps are normalised so the earliest-firing bin = 0 ms,
    then averaged across beats.

    Returns
    -------
    mean_map_ms : (H_bin, W_bin)  — mean activation delay in ms relative to
                                    the first-firing region; NaN for inactive bins.
    """
    H_bin, W_bin = activity_mask.shape
    N            = H_bin * W_bin
    mask_flat    = activity_mask.flatten()
    active_idx   = np.where(mask_flat)[0]

    if len(beat_peaks) == 0 or len(active_idx) == 0:
        return np.full((H_bin, W_bin), np.nan)

    lookback = int(fps_eff * 3.0)   # search up to 3 s before each peak

    beat_maps = []
    for peak_idx in beat_peaks:
        start     = max(0, peak_idx - lookback)
        act_times = np.full(N, np.nan)

        for i in active_idx:
            seg = corrected_signals[i, start: peak_idx + 1]
            if len(seg) < 2:
                continue

            peak_val  = seg[-1]
            baseline  = np.min(seg)
            amp       = peak_val - baseline
            if amp < 1e-6:
                continue

            threshold = baseline + 0.5 * amp
            matches   = np.where(seg >= threshold)[0]
            if len(matches) == 0:
                continue

            i_cross = matches[0]
            if i_cross == 0:
                act_times[i] = time_stamps[start]
            else:
                y1, y2 = seg[i_cross - 1], seg[i_cross]
                t1     = time_stamps[start + i_cross - 1]
                t2     = time_stamps[start + i_cross]
                if y2 > y1:
                    frac          = (threshold - y1) / (y2 - y1)
                    act_times[i]  = t1 + frac * (t2 - t1)
                else:
                    act_times[i] = t1

        valid = act_times[~np.isnan(act_times)]
        if len(valid) > 0:
            act_times -= np.nanmin(act_times)   # normalise: first-firing region = 0

        beat_maps.append(act_times.reshape(H_bin, W_bin))

    mean_map    = np.nanmean(np.array(beat_maps), axis=0)
    mean_map_ms = mean_map * 1000.0   # convert s → ms
    return mean_map_ms


def extract_spatiotemporal_features(corrected_signals, filtered_traces,
                                     activity_mask, beat_peaks, time_stamps, fps_eff):
    """
    Build a 5-feature descriptor per active bin:
      [pulsatility_score, dominant_freq, mean_beat_amplitude,
       mean_phase_offset, duty_cycle]

    Returns
    -------
    features   : (n_active, 5) float64
    active_idx : (n_active,)   indices into flattened H*W space
    """
    mask_flat  = activity_mask.flatten()
    active_idx = np.where(mask_flat)[0]

    if len(active_idx) == 0:
        return np.array([]), active_idx

    T     = corrected_signals.shape[1]
    dt    = (time_stamps[-1] - time_stamps[0]) / max(T - 1, 1)
    freqs = np.fft.rfftfreq(T, d=dt)
    band  = (freqs >= 0.3) & (freqs <= 5.0)
    win   = max(int(fps_eff * 1.5), 3)

    features = []
    for i in active_idx:
        sig  = corrected_signals[i]
        filt = filtered_traces[i]

        # 1. Pulsatility score
        puls = float(np.var(filt))

        # 2. Dominant frequency in cardiac band
        fft_mag = np.abs(np.fft.rfft(filt))
        dom_freq = float(freqs[band][np.argmax(fft_mag[band])]) \
                   if np.any(band) and np.any(fft_mag[band] > 0) else 0.0

        # 3. Mean beat amplitude
        if len(beat_peaks) > 0:
            amps = [float(sig[p] - np.min(sig[max(0, p - win): p + 1])) for p in beat_peaks]
            mean_amp = float(np.mean(amps))
        else:
            mean_amp = float(np.max(sig) - np.min(sig))

        # 4. Mean instantaneous phase (Hilbert of bandpassed signal)
        try:
            phase_offset = float(np.mean(np.angle(hilbert(filt))))
        except Exception:
            phase_offset = 0.0

        # 5. Duty cycle — fraction of time above 50% amplitude
        sr = np.max(sig) - np.min(sig)
        duty = float(np.mean(sig > (np.min(sig) + 0.5 * sr))) if sr > 1e-6 else 0.0

        features.append([puls, dom_freq, mean_amp, phase_offset, duty])

    return np.array(features, dtype=np.float64), active_idx


def cluster_spatiotemporal(features, active_idx, total_bins):
    """
    HDBSCAN on normalised spatiotemporal features.
    Falls back to KMeans(k=4) if hdbscan is not installed.

    Label conventions in returned array (length = total_bins):
      -2  inactive (not in activity mask)
      -1  HDBSCAN noise (active but unclustered)
       0, 1, 2 …  cluster IDs
    """
    labels_full = np.full(total_bins, -2, dtype=int)

    if len(features) < 3:
        return labels_full

    # Remove zero-variance columns before scaling (prevents NaN from StandardScaler)
    col_std = np.std(features, axis=0)
    good_cols = col_std > 1e-10
    if not np.any(good_cols):
        # All features identical → assign all active bins to cluster 0
        labels_full[active_idx] = 0
        return labels_full

    feat_valid  = features[:, good_cols]
    feat_scaled = StandardScaler().fit_transform(feat_valid)

    # Replace any remaining NaN/Inf (edge case) with 0
    feat_scaled = np.nan_to_num(feat_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        import hdbscan
        min_size  = max(3, len(feat_scaled) // 15)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, min_samples=2)
        labels    = clusterer.fit_predict(feat_scaled)
    except ImportError:
        from sklearn.cluster import KMeans
        n_c    = min(4, max(2, len(feat_scaled) // 5))
        labels = KMeans(n_clusters=n_c, n_init=10, random_state=42).fit_predict(feat_scaled)

    labels_full[active_idx] = labels
    return labels_full


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

class AnalysisWorker(QThread):
    finished = Signal(dict)
    error    = Signal(str)
    progress = Signal(int)

    def __init__(self, file_path, params):
        super().__init__()
        self.path   = file_path
        self.params = params

    def run(self):
        try:
            self.progress.emit(10)
            raw_stack = load_image(self.path)

            if raw_stack.ndim == 2:
                raw_stack = raw_stack[np.newaxis, ...]

            T_raw, H_raw, W_raw = raw_stack.shape

            is_fps = self.params.get('use_fps', False)
            if is_fps:
                fps            = self.params.get('val', 10.0)
                total_duration = T_raw / fps
            else:
                total_duration = self.params.get('val', 30.0)
                fps            = T_raw / total_duration

            sample = raw_stack[:min(16, T_raw)]
            hmin, hmax = np.percentile(sample, [0.4, 99.6])
            if hmax <= hmin:        # flat / constant image guard
                hmax = hmin + 1.0
            raw_stack  = np.clip(raw_stack, hmin, hmax)
            raw_stack  = (raw_stack - hmin) / (hmax - hmin) * 255.0

            self.progress.emit(20)

            b_size = self.params['binSize']
            sigma  = H_raw / 204.8
            frames = [block_reduce(gaussian_filter(f, sigma), (b_size, b_size), np.mean)
                      for f in raw_stack]
            binned_stack = np.array(frames)   # (T_bin, H_bin, W_bin)

            T_bin, H_bin, W_bin = binned_stack.shape
            fps_bin     = T_bin / total_duration
            time_stamps = np.linspace(0, total_duration, T_bin)

            self.progress.emit(30)

            # (1) Photobleaching correction
            raw_signals       = binned_stack.reshape(T_bin, -1).T   # (N, T_bin)
            corrected_signals = np.zeros_like(raw_signals)

            for i, sig in enumerate(raw_signals):
                try:
                    if self.params['model'] == 'Single Exp':
                        env = _bleach_envelope(sig, T_bin)
                        try:
                            a0   = max(float(env[0] - env[-1]), 1e-3)
                            p0   = [a0, total_duration / 3.0, float(env[-1])]
                            popt, _ = curve_fit(
                                _single_exp, time_stamps, env, p0=p0, maxfev=2000,
                                bounds=([0, 1e-5, -np.inf], [np.inf, np.inf, np.inf]),
                            )
                            baseline = _single_exp(time_stamps, *popt)
                        except Exception:
                            baseline = env
                    else:
                        min_idx, _ = find_peaks(-sig, distance=max(5, len(sig) // 10))
                        if len(min_idx) > 1:
                            f_i = interp1d(time_stamps[min_idx], sig[min_idx],
                                           kind='linear', fill_value="extrapolate")
                            baseline = f_i(time_stamps)
                        else:
                            baseline = _bleach_envelope(sig, T_bin)
                    corrected_signals[i] = sig - baseline
                except Exception:
                    corrected_signals[i] = sig - np.min(sig)

            self.progress.emit(50)

            # Guard: if corrected signals are all NaN (corrupt file / bad correction)
            nan_frac = np.mean(~np.isfinite(corrected_signals))
            if nan_frac > 0.9:
                raise ValueError(
                    "Corrected signals are mostly NaN/Inf. "
                    "Check that the file has valid pixel values and FPS is set correctly."
                )
            # Replace any remaining NaN with 0 to avoid cascade failures
            corrected_signals = np.nan_to_num(corrected_signals, nan=0.0, posinf=0.0, neginf=0.0)

            # (2) Pulsatility map + activity mask
            pulsatility_map, filtered_traces = compute_pulsatility_map(
                corrected_signals, H_bin, W_bin, T_bin, fps_bin
            )
            activity_mask = compute_activity_mask(pulsatility_map)

            # (3) Reference trace + beat detection
            reference_trace = compute_reference_trace(corrected_signals, activity_mask)
            beat_peaks      = detect_beats(reference_trace, time_stamps, fps_bin)

            self.progress.emit(65)

            # (4) Activation time wave map
            activation_map = compute_activation_time_map(
                corrected_signals, activity_mask, beat_peaks, time_stamps, fps_bin
            )

            # (5) Spatiotemporal features + HDBSCAN
            features, active_idx = extract_spatiotemporal_features(
                corrected_signals, filtered_traces, activity_mask,
                beat_peaks, time_stamps, fps_bin
            )

            self.progress.emit(80)

            n_total    = H_bin * W_bin
            labels_full = cluster_spatiotemporal(features, active_idx, n_total)

            # Cluster map for napari:
            #   0 = inactive background
            #   1 = active but HDBSCAN noise (unclustered)
            #   2+ = clusters 0, 1, 2 …
            clu_map = np.zeros(n_total, dtype=int)
            for idx in active_idx:
                lbl         = labels_full[idx]
                clu_map[idx] = 1 if lbl == -1 else lbl + 2
            clu_map = clu_map.reshape(H_bin, W_bin)

            active_sigs = corrected_signals[active_idx] if len(active_idx) > 0 \
                          else corrected_signals[:0]
            sync_index  = calculate_synchronicity(active_sigs) if len(active_idx) > 1 else 0.0

            self.progress.emit(100)

            self.finished.emit({
                'clu_map':           clu_map,
                'labels':            labels_full,
                'corrected_signals': corrected_signals,
                'filtered_traces':   filtered_traces,
                'time':              time_stamps,
                'dims':              (H_bin, W_bin),
                'activity_mask':     activity_mask,
                'pulsatility_map':   pulsatility_map,
                'activation_map':    activation_map,
                'beat_peaks':        beat_peaks,
                'reference_trace':   reference_trace,
                'sync_index':        sync_index,
                'beat_count':        len(beat_peaks),
            })

        except Exception as e:
            print("\n CRASH IN BACKGROUND THREAD:")
            traceback.print_exc()
            self.error.emit(str(e))


class BatchWorker(QThread):
    finished      = Signal(object)
    error         = Signal(str)
    progress      = Signal(int)
    file_progress = Signal(str)

    def __init__(self, file_paths, params):
        super().__init__()
        self.file_paths = file_paths
        self.params     = params

    def run(self):
        all_rows   = []
        total_files = len(self.file_paths)
        samples_per_file = self.params.get('batch_samples', 10)

        for f_idx, path in enumerate(self.file_paths):
            try:
                fname = os.path.basename(path)
                self.file_progress.emit(f"Processing {f_idx + 1}/{total_files}: {fname}")

                raw_stack = load_image(path)
                if raw_stack.ndim == 2:
                    raw_stack = raw_stack[np.newaxis, ...]

                fps    = self.params.get('val', 10.0)
                is_fps = self.params.get('use_fps', True)
                if path.lower().endswith(('.tif', '.tiff')):
                    try:
                        with tifffile.TiffFile(path) as tif:
                            ij_meta = tif.imagej_metadata or {}
                            if 'finterval' in ij_meta and ij_meta['finterval'] > 0:
                                fps    = 1.0 / ij_meta['finterval']
                                is_fps = True
                    except Exception:
                        pass

                T_raw = raw_stack.shape[0]
                total_duration = T_raw / fps if is_fps else self.params.get('val', 30.0)

                T, H, W = raw_stack.shape
                b_size  = 32 if max(H, W) >= 1024 else 16

                sample = raw_stack[:min(16, T)]
                hmin, hmax = np.percentile(sample, [0.4, 99.6])
                raw_stack  = np.clip(raw_stack, hmin, hmax)
                raw_stack  = (raw_stack - hmin) / (hmax - hmin) * 255.0

                sigma  = H / 204.8
                frames = [block_reduce(gaussian_filter(f, sigma), (b_size, b_size), np.mean)
                          for f in raw_stack]
                binned_stack = np.array(frames)

                T_bin, H_bin, W_bin = binned_stack.shape
                fps_bin     = T_bin / total_duration
                time_stamps = np.linspace(0, total_duration, T_bin)

                raw_signals       = binned_stack.reshape(T_bin, -1).T
                corrected_signals = np.zeros_like(raw_signals)

                for i, sig in enumerate(raw_signals):
                    try:
                        min_idx, _ = find_peaks(-sig, distance=max(5, len(sig) // 10))
                        if len(min_idx) > 1:
                            f_i = interp1d(time_stamps[min_idx], sig[min_idx],
                                           kind='linear', fill_value="extrapolate")
                            baseline = f_i(time_stamps)
                        else:
                            baseline = _bleach_envelope(sig, T_bin)
                        corrected_signals[i] = sig - baseline
                    except Exception:
                        corrected_signals[i] = sig - np.min(sig)

                # Use pulsatility mask for active bin selection
                pulsatility_map, _ = compute_pulsatility_map(
                    corrected_signals, H_bin, W_bin, T_bin, fps_bin
                )
                activity_mask = compute_activity_mask(pulsatility_map)
                active_idx    = np.where(activity_mask.flatten())[0]

                if len(active_idx) == 0:
                    continue

                valid_sigs = corrected_signals[active_idx]
                amps       = np.max(valid_sigs, axis=1) - np.min(valid_sigs, axis=1)
                weights    = amps ** 2
                probs      = weights / np.sum(weights) if np.sum(weights) > 0 else None

                n_choose      = min(samples_per_file, len(active_idx))
                chosen_indices = np.random.choice(active_idx, size=n_choose, replace=False, p=probs)

                for idx in chosen_indices:
                    m = extract_detailed_features(time_stamps, corrected_signals[idx])
                    if m:
                        y, x = divmod(int(idx), W_bin)
                        m.update({
                            'Filename':    fname,
                            'X (Binned)': x,
                            'Y (Binned)': y,
                            'ID':          idx,
                        })
                        all_rows.append(m)

                self.progress.emit(int(((f_idx + 1) / total_files) * 100))

            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        df = pd.DataFrame(all_rows)
        if not df.empty:
            priority = ['Filename', 'ID', 'X (Binned)', 'Y (Binned)']
            rest     = [c for c in df.columns if c not in priority]
            df       = df[priority + rest]

        self.finished.emit(df)
