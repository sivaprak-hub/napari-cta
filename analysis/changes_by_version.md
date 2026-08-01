# napari-cta — Changes by Version

---

## v0.1 — Initial Release (2026-06-16, commit 1cc4a59)

### backend.py
- ETS (Olympus VSI) binary reader
- Image loading: TIFF and VSI via aicsimageio
- Spatial binning with block_reduce
- Photobleach correction: Single Exp and Boundary models
- Pulsatility map with bandpass filter
- K-Means clustering of waveform features
- Basic kinetic extraction: extract_detailed_features()
  - T_ON, T_OFF, T10/T50/T90 ON/OFF
  - CD = duration above 10% amplitude (OLD formula)
- AnalysisWorker QThread

### widget.py
- CalciumControls: file queue, bin size, model, FPS/duration controls
- ResultsWidget: matplotlib graph + metrics table
- Basic file add/remove
- Run Analysis button
- Export to Excel (single file)

---

## v0.2 — FPS + Kinetics Overhaul (2026-07-09, commit 3dc2018)

### backend.py changes
- NEW: save_fps_sidecar() — saves .fps file alongside TIFF for persistence
- NEW: read_file_timing() — reads FPS from sidecar or tifffile metadata
- NEW: _fit_decay_tau() — log-linear exponential decay fit for stable T_OFF
- NEW: extract_beat_averaged_features() — per-beat kinetics, then averages
  - Separate end_search (1.5× period) and end_tau (valley-clipped)
  - T90_OFF and T_OFF via exp fit: τ·ln(10) and τ·ln(20)
  - CD = T_ON_ms + T_OFF_ms (updated formula)
- FIX: extract_detailed_features() — no-valley fallback uses 5% threshold
- FIX: Decay search extended to 1.5× beat period (prevents T90_OFF NaN)

### widget.py changes
- NEW: Results caching per (file, binSize, model, mode, val)
- NEW: FPS validation warning when metadata absent
- NEW: Auto-saves FPS sidecar after first run
- NEW: CollapsibleSection widget for organized UI
- FIX: Export traces layout (time as rows, cells as columns)

---

## v0.3 — Per-File State Persistence (2026-07-19, commit bfe8bce)

### widget.py changes
- NEW: _file_ui_state dict — stores layers, scores, chart, table per file
- NEW: _save_ui_state() / _restore_ui_state() on file switch
- Layers (cluster map, pulsatility overlay) restored when switching files

---

## v0.4 — Raw Signal F0 + Theme (2026-07-24, commits dfe0e71 + 9103784)

### backend.py changes
- FIX: F0 now from raw_signal (pre-correction), not corrected signal
- FIX: Valley detection uses find_peaks(-post_seg) not argmin
- FIX: Decay window split: end_search for thresholds, end_tau for exp fit
- FIX: Fallback kinetics via monoexp ratios when thresholds fail
- FIX: CD simplified to T_ON + T_OFF (no T10_ON subtraction)
- FIX: Sync index = 1.0 for single-cell recordings (not 0.0)

### widget.py changes
- NEW: Full dark theme stylesheet (_PANEL_STYLE_DARK)
- NEW: Detachable graph window (pop-out button)
- NEW: Layer visibility toggles (cluster map, pulsatility)
- NEW: Per-file state includes chart data and table rows
- FIX: BPM label now shows estimated BPM alongside beat count

---

## v0.5 — Bug Fixes + White Graph (2026-07-29, commit 37ecd5f)

### backend.py changes
- FIX: T_OFF decay window contamination bug — per-cell valley clipping
- FIX: CD formula finalised as T_ON + T_OFF in both functions
- FIX: Docstrings updated to match actual Ca²⁺ transient diagram

### widget.py changes
- NEW: Clear All button — removes all files from queue at once
- FIX: Peak markers now use per-cell find_peaks (not reference-trace)
- FIX: Graph background changed to white (#ffffff)
- FIX: QSpinBox button padding/positioning

---

## v0.6 — Light/Dark Theme Toggle (2026-07-31, commit f20f729)

### widget.py changes
- NEW: _PANEL_STYLE_LIGHT stylesheet (light panel theme)
- NEW: apply_theme() responds to napari theme change events
- FIX: Graph canvas always white regardless of panel theme
- FIX: Legend and axis colors use fixed dark-on-white palette
