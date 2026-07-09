import os
import numpy as np
import pandas as pd
import tifffile
import napari

import matplotlib
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QDoubleSpinBox, QComboBox, QGroupBox,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QProgressBar, QMessageBox, QSplitter, QListWidget, QListWidgetItem,
                             QCheckBox, QAbstractItemView, QScrollArea, QSizePolicy, QApplication)
from qtpy.QtCore import Qt, QTimer, QSize
from qtpy.QtGui import QColor


def _screen_geom():
    """Return (width, height) of the primary screen's available area."""
    app = QApplication.instance()
    if app is None:
        return 1920, 1080
    screen = app.primaryScreen()
    if screen is None:
        return 1920, 1080
    g = screen.availableGeometry()
    return g.width(), g.height()

from .backend import (AnalysisWorker, BatchWorker,
                       extract_detailed_features, extract_beat_averaged_features,
                       load_image, convert_single_vsi, read_file_timing,
                       save_fps_sidecar)


class VsiConverterWorker(AnalysisWorker.__bases__[0]):   # reuse QThread
    from qtpy.QtCore import Signal as _Signal
    progress     = _Signal(int)
    status       = _Signal(str)
    error_signal = _Signal(str)
    finished     = _Signal()

    def __init__(self, files):
        super().__init__()
        self.files = files

    def run(self):
        total = len(self.files)
        for i, f in enumerate(self.files):
            success, message = convert_single_vsi(f)
            if success:
                self.status.emit(f"Converted: {message}")
            else:
                self.error_signal.emit(f"Failed to convert {os.path.basename(f)}:\n{message}")
            self.progress.emit(int(((i + 1) / total) * 100))
        self.finished.emit()


# Re-declare the worker with proper QThread inheritance (the above shortcut
# can't resolve QThread at class-body time reliably on all Qt bindings).
from qtpy.QtCore import QThread, Signal

class VsiConverterWorker(QThread):
    progress     = Signal(int)
    status       = Signal(str)
    error_signal = Signal(str)
    finished     = Signal()

    def __init__(self, files):
        super().__init__()
        self.files = files

    def run(self):
        total = len(self.files)
        for i, f in enumerate(self.files):
            success, message = convert_single_vsi(f)
            if success:
                self.status.emit(f"Converted: {message}")
            else:
                self.error_signal.emit(f"Failed to convert {os.path.basename(f)}:\n{message}")
            self.progress.emit(int(((i + 1) / total) * 100))
        self.finished.emit()


class CalciumControls(QWidget):
    """Left-panel controls — napari plugin widget."""

    def __init__(self, napari_viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self.viewer            = napari_viewer
        self.raw_stack         = None
        self.processed_results = None
        self.last_path         = None
        self.master_results    = []
        self.master_traces     = []
        self._fps_source       = None
        self._results_cache    = {}   # (path, bin, model, mode, val) → results dict

        # Outer widget wraps a scroll area so the panel works on small screens
        sw, _sh = _screen_geom()
        panel_w = max(200, sw // 8)
        self.setMinimumWidth(panel_w)
        self.setMaximumWidth(int(panel_w * 1.25))
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        inner_widget = QWidget()
        self.layout  = QVBoxLayout(inner_widget)
        self.layout.setContentsMargins(6, 6, 6, 6)
        self.layout.setSpacing(6)

        scroll.setWidget(inner_widget)
        outer.addWidget(scroll)

        self.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; "
            "border-radius: 4px; margin-top: 8px; padding-top: 4px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )

        # --- 1. FILE QUEUE ---
        g_queue  = QGroupBox("1. File Queue")
        l_queue  = QVBoxLayout()
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.setMaximumHeight(120)
        self.list_widget.setMinimumHeight(60)
        self.list_widget.currentItemChanged.connect(self.on_queue_item_changed)

        btn_layout     = QHBoxLayout()
        btn_add_files  = QPushButton("Add Files...")
        btn_add_files.clicked.connect(self.add_files_to_queue)
        btn_remove     = QPushButton("Remove Selected")
        btn_remove.setStyleSheet("color: #d32f2f;")
        btn_remove.clicked.connect(self.remove_selected_file)
        btn_layout.addWidget(btn_add_files)
        btn_layout.addWidget(btn_remove)

        self.chk_auto = QCheckBox("Auto-Process on Load")
        self.chk_auto.setChecked(True)

        l_queue.addLayout(btn_layout)
        l_queue.addWidget(self.list_widget)
        l_queue.addWidget(self.chk_auto)
        g_queue.setLayout(l_queue)

        # --- 2. PARAMETERS ---
        g_param = QGroupBox("2. Parameters")
        from qtpy.QtWidgets import QFormLayout
        l_param = QFormLayout()
        l_param.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        l_param.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        l_param.setContentsMargins(6, 4, 6, 4)
        self.combo_mode  = QComboBox(); self.combo_mode.addItems(["FPS", "Duration (s)"])
        self.spin_val    = QDoubleSpinBox(); self.spin_val.setValue(15.0); self.spin_val.setRange(0.1, 9999)
        self.spin_bin    = QSpinBox();       self.spin_bin.setValue(16);   self.spin_bin.setRange(2, 128)
        self.combo_model = QComboBox();      self.combo_model.addItems(["Single Exp", "Boundary"])
        self.lbl_frames  = QLabel("Frames: — | Duration: — s")
        self.lbl_frames.setWordWrap(True)
        self.lbl_frames.setStyleSheet("color: #90CAF9; font-size: 11px;")
        l_param.addRow("Mode:",   self.combo_mode)
        l_param.addRow("FPS/Dur:", self.spin_val)
        l_param.addRow("Bin size:", self.spin_bin)
        l_param.addRow("Baseline:", self.combo_model)
        l_param.addRow("Info:", self.lbl_frames)
        g_param.setLayout(l_param)

        self.combo_mode.currentTextChanged.connect(self._update_frame_info)
        self.spin_val.valueChanged.connect(self._update_frame_info)

        # --- 3. ANALYSIS ---
        g_act   = QGroupBox("3. Analysis")
        l_act   = QVBoxLayout()
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.start_analysis)
        self.btn_run.setEnabled(False)
        self.prog = QProgressBar()
        self.lbl_beats = QLabel("Beats detected: —")
        self.lbl_beats.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.lbl_sync  = QLabel("Sync index: —")
        self.lbl_sync.setStyleSheet("color: #90CAF9;")
        l_act.addWidget(self.btn_run)
        l_act.addWidget(self.prog)
        l_act.addWidget(self.lbl_beats)
        l_act.addWidget(self.lbl_sync)
        g_act.setLayout(l_act)

        # --- 4. GUIDED EXPORT ---
        g_export = QGroupBox("4. Guided Export")
        l_export = QVBoxLayout()
        self.btn_save_next = QPushButton("Verify, Save\n& Go Next")
        self.btn_save_next.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 5px;"
        )
        self.btn_save_next.clicked.connect(self.save_and_next)
        self.btn_save_next.setEnabled(False)
        self.lbl_master_count = QLabel("Verified Cells: 0")
        self.lbl_master_count.setStyleSheet("color: #4CAF50;")
        self.btn_export_master = QPushButton("Export Master\nExcel")
        self.btn_export_master.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold;"
        )
        self.btn_export_master.clicked.connect(self.export_master_data)
        l_export.addWidget(self.btn_save_next)
        l_export.addWidget(self.lbl_master_count)
        l_export.addWidget(self.btn_export_master)
        g_export.setLayout(l_export)

        # --- 5. VSI CONVERTER ---
        g_vsi = QGroupBox("5. VSI to TIFF Converter")
        l_vsi = QVBoxLayout()
        self.btn_vsi  = QPushButton("Convert VSI\nBatch...")
        self.btn_vsi.clicked.connect(self.convert_vsi_batch)
        self.lbl_vsi  = QLabel("Idle")
        self.prog_vsi = QProgressBar()
        l_vsi.addWidget(self.btn_vsi)
        l_vsi.addWidget(self.lbl_vsi)
        l_vsi.addWidget(self.prog_vsi)
        g_vsi.setLayout(l_vsi)

        self.layout.addWidget(g_queue)
        self.layout.addWidget(g_param)
        self.layout.addWidget(g_act)
        self.layout.addWidget(g_export)
        self.layout.addWidget(g_vsi)
        self.layout.addStretch(1)   # push all panels to the top

        # Create the bottom Traces & Metrics panel and attach it to the viewer
        self.results_widget = ResultsWidget(napari_viewer, controls=self)
        _dock = napari_viewer.window.add_dock_widget(
            self.results_widget, area='bottom', name='Traces & Metrics'
        )
        _dock.show()

    # ------------------------------------------------------------------
    # Queue & file loading
    # ------------------------------------------------------------------

    def _update_frame_info(self):
        if self.raw_stack is None:
            self.lbl_frames.setText("Frames: — | Duration: — s")
            self.lbl_frames.setStyleSheet("color: #90CAF9; font-size: 11px;")
            return
        T   = self.raw_stack.shape[0]
        val = self.spin_val.value()
        src = getattr(self, '_fps_source', None)
        if self.combo_mode.currentText() == "FPS":
            fps = val
            dur = T / fps if fps > 0 else 0
            tag = f"  ✓ {src}" if src else "  (manual)"
            self.lbl_frames.setText(f"{T} frames · {fps:.2f} fps · {dur:.1f} s{tag}")
            color = "#4CAF50" if src else "#90CAF9"
        else:
            dur = val
            fps = T / dur if dur > 0 else 0
            self.lbl_frames.setText(f"{T} frames · {fps:.2f} fps (calc) · {dur:.1f} s")
            color = "#90CAF9"
        self.lbl_frames.setStyleSheet(f"color: {color}; font-size: 11px;")

    def add_files_to_queue(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", "Bio-Formats (*.tif *.tiff *.vsi)"
        )
        if files:
            for f in files:
                item = QListWidgetItem(os.path.basename(f))
                item.setData(Qt.ItemDataRole.UserRole, f)
                self.list_widget.addItem(item)
            if self.list_widget.currentRow() == -1:
                self.list_widget.setCurrentRow(0)

    def on_queue_item_changed(self, current, previous):
        if not current:
            return
        self.load_file(current.data(Qt.ItemDataRole.UserRole))

    def load_file(self, fname):
        try:
            self.viewer.layers.clear()
            self.raw_stack = load_image(fname)

            if self.raw_stack.ndim == 2:
                self.raw_stack = self.raw_stack[np.newaxis, ...]

            self.viewer.add_image(self.raw_stack, name=os.path.basename(fname))
            self.last_path = fname
            self.btn_run.setEnabled(True)
            self.btn_save_next.setEnabled(True)

            T, H, W = self.raw_stack.shape
            timing = read_file_timing(fname)

            if timing['fps']:
                self.combo_mode.setCurrentText("FPS")
                self.spin_val.setValue(timing['fps'])
                self._fps_source = timing['source']
            else:
                self._fps_source = None
                # Validate current setting against this file's frame count.
                # If it gives an implausible duration, flag it loudly so the
                # user knows to correct it before running analysis.
                is_fps = (self.combo_mode.currentText() == "FPS")
                val    = self.spin_val.value()
                est_dur = T / max(val, 0.001) if is_fps else val
                if est_dur > 300 or est_dur < 1:
                    self.lbl_frames.setText(
                        f"⚠ No FPS in file metadata — current setting gives "
                        f"{est_dur:.0f}s for {T} frames. Please set FPS/Duration correctly!"
                    )
                    self.lbl_frames.setStyleSheet(
                        "color: #FF5722; font-size: 11px; font-weight: bold;"
                    )
                    self.spin_bin.setValue(16 if max(H, W) < 2048 else 32)
                    # Don't auto-process with obviously wrong timing
                    if self.chk_auto.isChecked():
                        QMessageBox.warning(
                            self, "FPS Not Detected",
                            f"No FPS metadata found in:\n{os.path.basename(fname)}\n\n"
                            f"Current setting gives {est_dur:.0f}s for {T} frames, which looks wrong.\n\n"
                            f"Please set the correct FPS or Duration and click 'Run Analysis'."
                        )
                    return
                # Duration looks plausible but still no metadata source
                self._update_frame_info()

            self.spin_bin.setValue(16 if max(H, W) < 2048 else 32)
            self._update_frame_info()

            if self.chk_auto.isChecked():
                self.start_analysis()

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Could not load file.\n\nError: {e}\n\nDid you install aicsimageio?"
            )

    def remove_selected_file(self):
        row = self.list_widget.currentRow()
        if row >= 0:
            self.list_widget.takeItem(row)
            if self.list_widget.count() == 0:
                self.viewer.layers.clear()
                self.raw_stack = None
                self.last_path = None
                self.btn_run.setEnabled(False)
                self.btn_save_next.setEnabled(False)
                if hasattr(self, 'results_widget'):
                    self.results_widget.clear_all()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def start_analysis(self):
        if self.raw_stack is None:
            return

        is_fps = (self.combo_mode.currentText() == "FPS")
        self._last_params = {
            'binSize': self.spin_bin.value(),
            'model':   self.combo_model.currentText(),
            'use_fps': is_fps,
            'val':     self.spin_val.value(),
        }

        cache_key = (
            self.last_path,
            self.spin_bin.value(),
            self.combo_model.currentText(),
            self.combo_mode.currentText(),
            round(self.spin_val.value(), 4),
        )

        if cache_key in self._results_cache:
            self.on_analysis_done(self._results_cache[cache_key])
            return

        self._pending_cache_key = cache_key
        self.btn_run.setEnabled(False)
        self.btn_save_next.setEnabled(False)
        self.prog.setValue(0)

        # Save user-entered FPS as sidecar so it auto-loads next time.
        # Only save when the value was NOT already read from embedded metadata.
        if self.last_path and not self._fps_source and is_fps:
            try:
                save_fps_sidecar(self.last_path, self.spin_val.value())
            except Exception:
                pass

        self.worker = AnalysisWorker(self.last_path, self._last_params)
        self.worker.progress.connect(self.prog.setValue)
        self.worker.finished.connect(self.on_analysis_done)
        self.worker.error.connect(lambda msg: QMessageBox.critical(self, "Analysis Error", msg))
        self.worker.start()

    def on_analysis_done(self, results):
        self.processed_results = results
        self.btn_run.setEnabled(True)
        self.btn_save_next.setEnabled(True)
        self.prog.setValue(100)

        if hasattr(self, '_pending_cache_key'):
            self._results_cache[self._pending_cache_key] = results
            del self._pending_cache_key

        self.lbl_beats.setText(f"Beats detected: {results['beat_count']}")
        self.lbl_sync.setText(f"Sync index: {results['sync_index']:.3f}")

        scale = (self.spin_bin.value(), self.spin_bin.value())

        # Remove stale layers
        for name in ['Clusters', 'Pulsatility', 'Wave Map', 'Selection']:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

        # Pulsatility map — intensity of periodic signal per bin
        self.viewer.add_image(
            results['pulsatility_map'],
            name='Pulsatility', scale=scale, opacity=0.5,
            colormap='inferno', blending='additive', visible=False,
        )

        # Activation time wave map — ms delay from first-firing region
        act_map = results['activation_map']
        if not np.all(np.isnan(act_map)):
            self.viewer.add_image(
                np.nan_to_num(act_map, nan=0.0),
                name='Wave Map', scale=scale, opacity=0.6,
                colormap='twilight_shifted', blending='additive',
            )

        # Cluster labels
        self.viewer.add_labels(results['clu_map'], name='Clusters', scale=scale, opacity=0.45)

        # Selection points layer for manual cell picking (pan_zoom so clicks don't auto-add napari points)
        pts = self.viewer.add_points(name='Selection', ndim=3, size=scale[0] * 2)
        pts.face_color = 'transparent'
        pts.edge_color = 'transparent'
        pts.mode       = 'pan_zoom'

        if hasattr(self, 'results_widget'):
            self.results_widget.set_data(results, self.spin_bin.value())
            if self.chk_auto.isChecked():
                self.results_widget.random_sample()

    # ------------------------------------------------------------------
    # Export workflow
    # ------------------------------------------------------------------

    def save_and_next(self):
        if not self.last_path or not hasattr(self, 'results_widget'):
            return
        filename = os.path.basename(self.last_path)
        metrics  = self.results_widget.get_current_metrics(filename)

        if not metrics:
            ans = QMessageBox.question(
                self, "No Points", "No points selected. Skip?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
                return

        self.master_results.extend(metrics)
        traces = self.results_widget.get_current_traces(filename)
        self.master_traces.extend(traces)
        self.lbl_master_count.setText(f"Verified Cells: {len(self.master_results)}")

        row = self.list_widget.currentRow()
        if row < self.list_widget.count() - 1:
            self.list_widget.setCurrentRow(row + 1)
        else:
            QMessageBox.information(self, "Done", "Queue finished! Export Master Excel now.")

    def export_master_data(self):
        if not self.master_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Master", "Master_Results.xlsx", "Excel (*.xlsx)"
        )
        if not path:
            return
        try:
            # --- Metrics sheet: fixed column order, one row per cell ---
            metric_col_order = [
                'Filename', 'Cell', 'X (Binned)', 'Y (Binned)',
                'BPM', 'Amp', 'F0',
                'T_ON_ms', 'T10_ON', 'T50_ON', 'T90_ON',
                'T_OFF_ms', 'T10_OFF', 'T50_OFF', 'T90_OFF',
                'CD',
            ]
            df_metrics = pd.DataFrame(self.master_results)
            # Keep only known columns, in order (ignore any extras)
            present = [c for c in metric_col_order if c in df_metrics.columns]
            df_metrics = df_metrics[present].round(3)

            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                df_metrics.to_excel(writer, sheet_name='Metrics', index=False)

                if self.master_traces:
                    # --- Traces sheet: rows = time points, columns = cells ---
                    # Build one column per cell; label = "File | Cell | Y,X"
                    time_ref = self.master_traces[0]['time']
                    df_traces = pd.DataFrame({'Time (s)': time_ref})
                    for tr in self.master_traces:
                        col_label = (
                            f"{os.path.splitext(tr['Filename'])[0]}"
                            f"_P{tr['Cell'].lstrip('P')}"
                            f"_Y{tr['Y']}X{tr['X']}"
                        )
                        # Pad / trim if time lengths differ between files
                        sig = tr['signal']
                        n   = len(df_traces)
                        if len(sig) < n:
                            sig = sig + [np.nan] * (n - len(sig))
                        df_traces[col_label] = sig[:n]
                    df_traces.to_excel(writer, sheet_name='Traces', index=False)

            n_sheets = 2 if self.master_traces else 1
            QMessageBox.information(
                self, "Success",
                f"Exported {len(df_metrics)} cells to {n_sheets} sheet(s)."
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    # ------------------------------------------------------------------
    # VSI conversion
    # ------------------------------------------------------------------

    def convert_vsi_batch(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select VSI Files", "", "VSI (*.vsi)")
        if not files:
            return
        self.lbl_vsi.setText("Initializing...")
        self.btn_vsi.setEnabled(False)
        self.conv_worker = VsiConverterWorker(files)
        self.conv_worker.progress.connect(self.prog_vsi.setValue)
        self.conv_worker.status.connect(self.lbl_vsi.setText)
        self.conv_worker.error_signal.connect(
            lambda msg: QMessageBox.critical(self, "Conversion Error", msg)
        )
        self.conv_worker.finished.connect(self._on_conversion_done)
        self.conv_worker.start()

    def _on_conversion_done(self):
        self.btn_vsi.setEnabled(True)
        self.lbl_vsi.setText("Conversion Complete!")
        QMessageBox.information(self, "Done", "Batch conversion finished.")


# ---------------------------------------------------------------------------
# Bottom panel — traces and metrics table
# ---------------------------------------------------------------------------

class ResultsWidget(QWidget):
    def __init__(self, viewer, controls, parent=None):
        super().__init__(parent)
        self.viewer   = viewer
        self.controls = controls
        self.results  = None
        self.bin_size = 1
        self.selected_coords = []

        _sw, sh = _screen_geom()
        self.setMinimumHeight(max(216, int(sh * 0.24)))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        import matplotlib.pyplot as plt
        cmap        = plt.get_cmap('tab20')
        self.colors = [matplotlib.colors.to_hex(cmap(i)) for i in np.linspace(0, 1, 50)]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Max Points:"))
        self.spin_max   = QSpinBox(); self.spin_max.setRange(1, 50); self.spin_max.setValue(6)
        self.btn_random = QPushButton("Random Sample")
        self.btn_random.clicked.connect(self.random_sample)
        self.btn_clear  = QPushButton("Clear Selection")
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_save   = QPushButton("Save Graph")
        self.btn_save.clicked.connect(self.save_graph)
        ctrl.addWidget(self.spin_max)
        ctrl.addWidget(self.btn_random)
        ctrl.addWidget(self.btn_clear)
        ctrl.addStretch()
        ctrl.addWidget(self.btn_save)
        layout.addLayout(ctrl)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.canvas = FigureCanvas(Figure())
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.figure.set_tight_layout(True)
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_xlabel("Time (s)", fontweight='bold')
        self.ax.set_ylabel("Amplitude (a.u.)", fontweight='bold')

        self.table = QTableWidget()
        cols = ["ID", "BPM", "Amp", "F0", "T_ON_ms", "T10_ON", "T50_ON", "T90_ON",
                "T_OFF_ms", "T10_OFF", "T50_OFF", "T90_OFF", "CD"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.table)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([1000, 1000])
        layout.addWidget(self.splitter)

        self.viewer.mouse_drag_callbacks.append(self.on_click)
        self.viewer.dims.events.current_step.connect(self.update_points_z)

    def set_data(self, results, bin_size):
        self.results  = results
        self.bin_size = bin_size
        self.selected_coords = []
        self.refresh_ui()

    def clear_all(self):
        self.selected_coords = []
        self.refresh_ui()

    def get_current_metrics(self, filename):
        rows = []
        if not self.results:
            return rows
        time       = self.results['time']
        sigs       = self.results['corrected_signals']
        W          = self.results['dims'][1]
        beat_peaks = self.results.get('beat_peaks', np.array([]))
        for i, (y, x) in enumerate(self.selected_coords):
            idx = y * W + x
            m   = extract_beat_averaged_features(time, sigs[idx], beat_peaks)
            if m:
                m.pop('CD_estimated', None)   # internal flag, not needed in export
                row = {
                    'Filename':   filename,
                    'Cell':       f'P{i + 1}',
                    'X (Binned)': x,
                    'Y (Binned)': y,
                    'BPM':        m.get('BPM'),
                    'Amp':        m.get('Amp'),
                    'F0':         m.get('F0'),
                    'T_ON_ms':    m.get('T_ON_ms'),
                    'T10_ON':     m.get('T10_ON'),
                    'T50_ON':     m.get('T50_ON'),
                    'T90_ON':     m.get('T90_ON'),
                    'T_OFF_ms':   m.get('T_OFF_ms'),
                    'T10_OFF':    m.get('T10_OFF'),
                    'T50_OFF':    m.get('T50_OFF'),
                    'T90_OFF':    m.get('T90_OFF'),
                    'CD':         m.get('CD'),
                }
                rows.append(row)
        return rows

    def get_current_traces(self, filename):
        """Return one dict per cell.  Time array + signal stored as lists so the
        exporter can pivot to a readable column-per-cell layout."""
        rows = []
        if not self.results:
            return rows
        time = self.results['time']
        sigs = self.results['corrected_signals']
        W    = self.results['dims'][1]
        for i, (y, x) in enumerate(self.selected_coords):
            idx = y * W + x
            rows.append({
                'Filename': filename,
                'Cell':     f'P{i + 1}',
                'X':        x,
                'Y':        y,
                'time':     list(time),
                'signal':   [float(v) for v in sigs[idx]],
            })
        return rows

    def random_sample(self):
        if not self.results:
            return
        limit  = self.spin_max.value()
        labels = self.results['labels']
        # Active bins: labels >= -1 (in activity mask; -2 = inactive background)
        active_idx = np.where(labels >= -1)[0]
        if len(active_idx) == 0:
            self.selected_coords = []
            self.refresh_ui()
            return

        sigs   = self.results['corrected_signals'][active_idx]
        amps   = np.max(sigs, axis=1) - np.min(sigs, axis=1)
        w      = amps ** 2
        probs  = w / np.sum(w) if np.sum(w) > 0 else None

        n      = min(limit, len(active_idx))
        chosen = np.random.choice(active_idx, size=n, replace=False, p=probs)

        W = self.results['dims'][1]
        self.selected_coords = [(int(idx // W), int(idx % W)) for idx in chosen]
        self.refresh_ui()

    def save_graph(self):
        if not self.results or not self.selected_coords:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Graph", "calcium_traces.png", "PNG Image (*.png)"
        )
        if path:
            self.canvas.figure.savefig(path, bbox_inches='tight', dpi=300)

    def on_click(self, viewer, event):
        if self.results is None:
            return
        if getattr(event, 'button', 0) != 1:   # left click only
            return
        c = viewer.cursor.position
        if len(c) == 3:
            _, y_raw, x_raw = c
        else:
            y_raw, x_raw = c
        y, x = int(y_raw / self.bin_size), int(x_raw / self.bin_size)
        H, W = self.results['dims']
        if not (0 <= x < W and 0 <= y < H):
            return
        pt = (y, x)
        if pt in self.selected_coords:
            self.selected_coords.remove(pt)
        else:
            self.selected_coords.append(pt)
            while len(self.selected_coords) > self.spin_max.value():
                self.selected_coords.pop(0)
        self.refresh_ui()

    def refresh_ui(self):
        self.ax.clear()
        self.table.setRowCount(0)
        self.ax.set_xlabel("Time (s)", fontweight='bold')
        self.ax.set_ylabel("Amplitude (a.u.)", fontweight='bold')
        self.ax.grid(True, alpha=0.3)

        if not self.results:
            self.canvas.draw()
            return

        time  = self.results['time']
        sigs  = self.results['corrected_signals']
        W     = self.results['dims'][1]
        t_idx = self.viewer.dims.current_step[0] if len(self.viewer.dims.current_step) > 2 else 0

        points_data = []
        face_colors = []
        text_labels = []

        beat_peaks = self.results.get('beat_peaks', np.array([]))

        for i, (y, x) in enumerate(self.selected_coords):
            idx   = y * W + x
            sig   = sigs[idx]
            color = self.colors[i % len(self.colors)]

            self.ax.plot(time, sig, color=color, label=f"P{i+1}", linewidth=1.2)

            # Mark detected beats on this trace
            if len(beat_peaks) > 0:
                self.ax.plot(time[beat_peaks], sig[beat_peaks], 'v',
                             color=color, markersize=5, alpha=0.7)

            m = extract_beat_averaged_features(time, sig, beat_peaks)
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(f"P{i+1}"))
            self.table.item(r, 0).setForeground(QColor(color))
            if m:
                keys = ['BPM', 'Amp', 'F0', 'T_ON_ms', 'T10_ON', 'T50_ON', 'T90_ON',
                        'T_OFF_ms', 'T10_OFF', 'T50_OFF', 'T90_OFF', 'CD']
                for c_idx, key in enumerate(keys):
                    val = m.get(key, np.nan)
                    txt = f"{val:.1f}" if not (isinstance(val, float) and np.isnan(val)) else "—"
                    self.table.setItem(r, c_idx + 1, QTableWidgetItem(txt))

            py = y * self.bin_size + self.bin_size / 2
            px = x * self.bin_size + self.bin_size / 2
            points_data.append([t_idx, py, px])
            face_colors.append(color)
            text_labels.append(f"P{i+1}")

        if self.selected_coords:
            self.ax.legend(loc='upper right', fontsize='small')
        self.canvas.draw()

        if 'Selection' in self.viewer.layers:
            layer = self.viewer.layers['Selection']
            if points_data:
                layer.data       = np.array(points_data)
                layer.face_color = face_colors
                layer.text       = {'string': text_labels, 'color': 'white',
                                    'translation': np.array([0, -5, 0])}
            else:
                layer.data = np.empty((0, 3))
            layer.refresh()

    def update_points_z(self, event):
        # Only shift the Z (time) coordinate of existing selection points.
        # Avoid a full matplotlib redraw + extract_detailed_features on every frame step.
        if not self.results or 'Selection' not in self.viewer.layers:
            return
        layer = self.viewer.layers['Selection']
        if len(layer.data) == 0:
            return
        t_idx = self.viewer.dims.current_step[0] if len(self.viewer.dims.current_step) > 2 else 0
        new_data = layer.data.copy()
        new_data[:, 0] = t_idx
        layer.data = new_data


# ---------------------------------------------------------------------------
# Entry point — run directly: python widget.py
# ---------------------------------------------------------------------------

def main():
    import napari
    viewer = napari.Viewer(title="Calcium Transient Analyzer")
    ctrl   = CalciumControls(viewer)          # ResultsWidget is created inside
    viewer.window.add_dock_widget(ctrl, area='right', name='CTA Controls')
    napari.run()


if __name__ == '__main__':
    main()
