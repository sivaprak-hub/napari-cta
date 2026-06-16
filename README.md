# napari-cta — Calcium Transient Analyzer

A napari plugin for automated analysis of fluorescence calcium imaging recordings from cardiomyocytes (or any excitable cells). Extracts spatial cluster maps and standard cardiac kinetic metrics — entirely within the napari viewer, no scripting required.

---

## Table of Contents
1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Project Structure](#3-project-structure)
4. [Quick Start](#4-quick-start)
5. [Interface Reference](#5-interface-reference)
6. [Parameters Reference](#6-parameters-reference)
7. [Signal Processing Pipeline](#7-signal-processing-pipeline)
8. [Output Data Reference](#8-output-data-reference)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Overview

### What CTA does
- Loads time-lapse fluorescence microscopy recordings (TIFF or Olympus VSI/ETS format)
- Bins and smooths the image stack to generate per-pixel calcium traces
- Corrects photobleach drift using baseline subtraction
- Clusters cells by waveform shape using **K-Means**; projects results as a colour-coded spatial map
- Extracts standard cardiac kinetic metrics: BPM, amplitude, rise/decay times, contraction duration
- Provides an interactive trace viewer and guided export workflow to Excel (Metrics + Traces sheets)

### Key features

| Feature | Detail |
|---------|--------|
| Supported formats | `.tif`, `.tiff`, `.vsi` (Olympus ETS) |
| Spatial binning | Configurable; auto-tunes to image size |
| Baseline correction | Single-exponential fit or valley interpolation |
| Clustering | K-Means on waveform features |
| Batch mode | Multi-file queue with amplitude²-weighted random sampling |
| Export | Two-sheet Excel: Metrics (kinetics) + Traces (raw signal per time point) |

---

## 2. Installation

### Prerequisites
- Python 3.9 or later
- napari 0.4 or later

### Install as a napari plugin
From the folder containing `pyproject.toml`:

```bash
pip install -e .
```

Then open napari — the plugin appears under **Plugins → CTA Controls**.

### Or run standalone
```bash
python -m CTA_Fixed.widget
```

### Dependencies (installed automatically)
```
napari, numpy, pandas, tifffile, scipy, scikit-image, scikit-learn,
matplotlib, openpyxl, aicsimageio, imagecodecs
```

---

## 3. Project Structure

```
napari-cta/
├── CTA_Fixed/
│   ├── __init__.py        # Plugin entry point
│   ├── backend.py         # Signal processing, ETS reader, feature extraction, worker threads
│   ├── widget.py          # napari UI — left control panel + bottom results panel
│   └── napari.yaml        # npe2 plugin manifest
└── pyproject.toml         # Package metadata and napari entry point
```

### Key classes

| Name | File | Description |
|------|------|-------------|
| `CalciumControls` | widget.py | Left panel widget; file queue, parameters, analysis controls |
| `ResultsWidget` | widget.py | Bottom panel; trace graph + metrics table |
| `AnalysisWorker` | backend.py | QThread; runs full pipeline for one file |
| `extract_detailed_features` | backend.py | Per-signal kinetic feature extraction |

---

## 4. Quick Start

1. Open napari: `python -m napari`
2. Go to **Plugins → CTA Controls**
3. Click **Add Files...** and select a `.tif` or `.vsi` file
4. With **Auto-Process on Load** checked, analysis starts immediately
5. A cluster colour map appears as an overlay on the image
6. Click any coloured region to select that cell — its trace and metrics appear in the bottom panel
7. Click **Verify, Save & Go Next** to save that file's cells and move to the next file
8. Click **Export Master Excel** when done with all files

---

## 5. Interface Reference

### Panel 1 — File Queue

| Control | Function |
|---------|----------|
| **Add Files...** | Opens file dialog; supports `.tif`, `.tiff`, `.vsi`. Multiple files accepted. |
| **Remove Selected** | Removes the highlighted file from the queue. |
| **File list** | Click any item to load that file immediately. |
| **Auto-Process on Load** | When checked, analysis runs automatically on load. |

### Panel 2 — Parameters

| Control | Function |
|---------|----------|
| **Mode** | Toggle between FPS (frames per second) or Duration (total seconds). |
| **FPS / Dur value** | Numeric value for the selected mode. Auto-filled from TIFF metadata when available. |
| **Bin size** | Spatial bin in pixels (e.g. 16 → 16×16 blocks). Auto-set to 16 for images ≤ 2047 px, 32 for larger. |
| **Baseline model** | `Single Exp`: exponential decay fit (better for long photobleach). `Boundary`: valley interpolation (faster). |
| **Info bar** | Shows frame count, FPS, and total duration for the loaded file. |

### Panel 3 — Analysis

| Control | Function |
|---------|----------|
| **Run Analysis** | Starts analysis with current parameters. Available when a file is loaded. |
| **Progress bar** | Tracks pipeline progress from load → bin → correct → cluster. |
| **Beats detected** | Beat count from the averaged trace after analysis. |
| **Sync index** | Mean pairwise Pearson correlation across active cells (0 = no sync, 1 = perfect sync). |

### Panel 4 — Guided Export

| Control | Function |
|---------|----------|
| **Verify, Save & Go Next** | Saves selected cells from current file into master list, advances to next file. |
| **Verified Cells** | Running count of saved cells across all files. |
| **Export Master Excel** | Saves all verified cells to a `.xlsx` with two sheets: Metrics and Traces. |

### Panel 5 — VSI to TIFF Converter

| Control | Function |
|---------|----------|
| **Convert VSI Batch...** | Converts one or more Olympus `.vsi` files to 16-bit TIFF. |

### Bottom Panel — Traces & Metrics

| Control | Function |
|---------|----------|
| **Max Points** | Maximum number of simultaneously shown traces (1–50). |
| **Random Sample** | Randomly selects cells, biased toward high-amplitude signals. |
| **Clear Selection** | Removes all selected cells from graph and table. |
| **Save Graph** | Saves the current trace plot as a 300 dpi PNG. |
| **Click on image** | Clicking a pixel adds/removes that cell's trace. Selected cells show as numbered dots on the cluster map. |

---

## 6. Parameters Reference

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| FPS | 15.0 | 0.1 – 9999 | Auto-read from TIFF metadata when available |
| Duration | 15.0 | 0.1 – 9999 s | Alternative to FPS |
| Bin size | 16 px | 2 – 128 | Larger = fewer cells, faster; smaller = more cells, slower |
| Baseline model | Single Exp | — | Single Exp handles photobleach better; Boundary is faster |
| Max Points | 6 | 1 – 50 | Maximum concurrent traces in bottom panel |

---

## 7. Signal Processing Pipeline

### Step 1 — Image loading
- **TIFF**: `tifffile.imread` → `(T, H, W)` array; FPS read from ImageJ metadata
- **VSI (Olympus)**: Direct ETS binary reader (no aicsimageio required for the main path); fallback to `aicsimageio`
- 2D single frames promoted to `(1, H, W)`

### Step 2 — Intensity normalisation
- Sample 16 frames to compute 0.4th and 99.6th percentile
- Clip to this range and rescale to `[0, 255]`
- Removes hot pixels and outlier frames

### Step 3 — Spatial binning and smoothing
- Gaussian blur (σ = H/204.8) per frame
- `skimage.measure.block_reduce` averages `(bin × bin)` tiles
- Output: `(N_cells, T)` signal matrix

### Step 4 — Baseline correction
For each trace:
1. Detect valley anchor points
2. Interpolate baseline through valleys
3. If **Single Exp**: fit `a·exp(−t/τ) + c` to baseline; subtract fitted curve
4. If **Boundary**: subtract interpolated valley curve directly

### Step 5 — Active cell detection
A cell is active if its corrected signal range exceeds 0.5 intensity units.

### Step 6 — K-Means clustering
- Features: mean, std, max, skewness, kurtosis of each trace
- `sklearn.cluster.KMeans` (n_init=10) fitted to active cell features
- Inactive cells receive label −1
- Labels reshaped to `(H_bin, W_bin)` and displayed as a napari Labels layer

### Step 7 — Synchronicity index
Mean off-diagonal Pearson correlation across all active cell traces. Values near 1.0 = all cells beat in phase.

---

## 8. Output Data Reference

### Master Excel — Metrics sheet

| Column | Unit | Description |
|--------|------|-------------|
| Filename | — | Source file name |
| ID | — | Linear cell index |
| X (Binned) | px | Column in binned coordinates |
| Y (Binned) | px | Row in binned coordinates |
| BPM | beats/min | Estimated beat rate |
| Amp | a.u. | Peak amplitude above baseline |
| F0 | a.u. | Resting baseline fluorescence |
| T_ON_ms | ms | Time from transient start to peak |
| T10_ON | ms | Rise time to 10% of peak |
| T50_ON | ms | Rise time to 50% of peak |
| T90_ON | ms | Rise time to 90% of peak |
| T_OFF_ms | ms | Time from peak to transient end |
| T10_OFF | ms | Decay time to 90% level |
| T50_OFF | ms | Half-decay time |
| T90_OFF | ms | Decay time to 10% level |
| CD | ms | Contraction duration |

### Master Excel — Traces sheet

One row per selected cell. Columns: `Filename`, `ID`, `X (Binned)`, `Y (Binned)`, followed by one column per time point labelled `t=X.XXXs`.

---

## 9. Troubleshooting

### "Could not load file" on a VSI file
Install the optional aicsimageio fallback:
```bash
pip install aicsimageio
```

### Analysis produces no clusters / all cells inactive
- Verify FPS or Duration is correct for the recording
- Try reducing Bin size (e.g. 32 → 16) to increase cell count
- Check that the recording actually contains beating cells

### Bottom panel not visible
The bottom panel is added automatically when the plugin loads. If it is hidden, go to **Window** in napari's menu bar and enable **Traces & Metrics**.

### Plugin not visible in Plugins menu
Re-run the editable install from the folder containing `pyproject.toml`:
```bash
pip install -e .
```

### VSI conversion produces wrong frame intervals
The converter reads the frame interval from VSI metadata. If not found, it defaults to 1 fps. Correct the FPS manually in Panel 2 after loading the converted TIFF.

---

*Developed at Trinity College Dublin*
