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
                             QSpinBox, QDoubleSpinBox, QComboBox, QAction,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QProgressBar, QMessageBox, QSplitter, QListWidget, QListWidgetItem,
                             QCheckBox, QAbstractItemView, QScrollArea, QSizePolicy, QApplication)
from qtpy.QtCore import Qt, QTimer, QSize, QThread, Signal
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


class CollapsibleSection(QWidget):
    """A titled section that can be expanded or collapsed by clicking its header."""

    def __init__(self, title, parent=None, expanded=True):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 2)
        root.setSpacing(0)

        self._title = title

        self._btn = QPushButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.setStyleSheet(
            "QPushButton {"
            "  text-align: left; font-weight: bold; padding: 4px 6px;"
            "  background: #3c3c3c; border: 1px solid #555; border-radius: 3px;"
            "}"
            "QPushButton:hover { background: #484848; }"
        )
        self._btn.toggled.connect(self._on_toggled)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 4, 4, 4)
        self._body_layout.setSpacing(4)
        self._body.setVisible(expanded)

        root.addWidget(self._btn)
        root.addWidget(self._body)
        self._refresh_label(expanded)

    def _refresh_label(self, expanded):
        icon = "▼" if expanded else "▶"
        self._btn.setText(f"  {icon}  {self._title}")

    def _on_toggled(self, checked):
        self._body.setVisible(checked)
        self._refresh_label(checked)

    @property
    def body_layout(self):
        return self._body_layout

    def is_expanded(self):
        return self._btn.isChecked()

    def set_expanded(self, val: bool):
        self._btn.setChecked(val)


class CalciumControls(QWidget):
    """Left-panel controls — napari plugin widget."""

    # Bug 5: class-level registry prevents duplicate bottom panels across re-opens.
    _results_docks = {}   # id(viewer) → results dock widget

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
        self._verified_paths   = set()  # Bug 8: track which files have been saved
        self._file_states      = {}   # UI #2: path → 'unprocessed'|'analysed'|'verified'
        self._file_ui_state    = {}   # full UI state per file for lossless switching

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

        # --- Collapse All / Expand All button ---
        self._collapsed_all = False
        self._btn_collapse_all = QPushButton("⊟  Collapse All")
        self._btn_collapse_all.setStyleSheet(
            "font-size: 10px; padding: 2px 6px; background: #2a2a2a; border: 1px solid #444;"
        )
        self._btn_collapse_all.clicked.connect(self._toggle_collapse_all)
        self.layout.addWidget(self._btn_collapse_all)

        # --- 1. FILE QUEUE ---
        sec_queue = CollapsibleSection("1. File Queue")
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
        self.chk_auto.setChecked(False)

        sec_queue.body_layout.addLayout(btn_layout)
        sec_queue.body_layout.addWidget(self.list_widget)
        sec_queue.body_layout.addWidget(self.chk_auto)

        # --- 2. PARAMETERS ---
        sec_param = CollapsibleSection("2. Parameters")
        from qtpy.QtWidgets import QFormLayout
        l_param = QFormLayout()
        l_param.setRowWrapPolicy(QFormLayout.RowWrapPolicy.DontWrapRows)
        l_param.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        l_param.setContentsMargins(2, 2, 2, 2)
        self.combo_mode  = QComboBox(); self.combo_mode.addItems(["FPS", "Duration (s)"]); self.combo_mode.setCurrentText("Duration (s)")
        self.spin_val    = QDoubleSpinBox(); self.spin_val.setValue(15.0); self.spin_val.setRange(0.1, 9999)
        self.spin_bin    = QSpinBox();       self.spin_bin.setValue(16);   self.spin_bin.setRange(2, 128)
        self.combo_model = QComboBox();      self.combo_model.addItems(["Single Exp", "Boundary"])
        # UI #1: dynamic label that changes with mode selection
        self.lbl_val = QLabel("Duration (s):")
        l_param.addRow("Mode:",        self.combo_mode)
        l_param.addRow(self.lbl_val,   self.spin_val)
        l_param.addRow("Bin size:",    self.spin_bin)
        l_param.addRow("Baseline:",    self.combo_model)
        sec_param.body_layout.addLayout(l_param)

        # Info label sits below the form so it gets full width and wraps properly
        self.lbl_frames = QLabel("Frames: — | Duration: — s")
        self.lbl_frames.setWordWrap(True)
        self.lbl_frames.setStyleSheet("color: #90CAF9; font-size: 11px;")
        sec_param.body_layout.addWidget(self.lbl_frames)

        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        self.spin_val.valueChanged.connect(self._update_frame_info)

        # --- 3. ANALYSIS ---
        sec_act = CollapsibleSection("3. Analysis")
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.start_analysis)
        self.btn_run.setEnabled(False)
        self.prog = QProgressBar()
        self.lbl_beats = QLabel("Beats detected: —")
        self.lbl_beats.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.lbl_sync  = QLabel("Sync index: —")
        self.lbl_sync.setStyleSheet("color: #90CAF9;")
        # Bug 4: button to restore the bottom panel if the user accidentally closes it
        self.btn_show_traces = QPushButton("Show Traces Panel")
        self.btn_show_traces.setStyleSheet("font-size: 11px;")
        self.btn_show_traces.clicked.connect(self._show_results_panel)
        sec_act.body_layout.addWidget(self.btn_run)
        sec_act.body_layout.addWidget(self.prog)
        sec_act.body_layout.addWidget(self.lbl_beats)
        sec_act.body_layout.addWidget(self.lbl_sync)
        sec_act.body_layout.addWidget(self.btn_show_traces)

        # --- 4. GUIDED EXPORT ---
        sec_export = CollapsibleSection("4. Guided Export")
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
        self.btn_export_master.setEnabled(False)  # UI #3: disabled until cells are verified
        sec_export.body_layout.addWidget(self.btn_save_next)
        sec_export.body_layout.addWidget(self.lbl_master_count)
        sec_export.body_layout.addWidget(self.btn_export_master)

        # --- 5. VSI CONVERTER (collapsed by default) ---
        sec_vsi = CollapsibleSection("5. VSI to TIFF Converter", expanded=False)
        self.btn_vsi  = QPushButton("Convert VSI\nBatch...")
        self.btn_vsi.clicked.connect(self.convert_vsi_batch)
        self.lbl_vsi  = QLabel("Idle")
        self.prog_vsi = QProgressBar()
        sec_vsi.body_layout.addWidget(self.btn_vsi)
        sec_vsi.body_layout.addWidget(self.lbl_vsi)
        sec_vsi.body_layout.addWidget(self.prog_vsi)

        # VSI is always collapsed; exclude it from Collapse/Expand All
        self._all_sections = [sec_queue, sec_param, sec_act, sec_export]

        self.layout.addWidget(sec_queue)
        self.layout.addWidget(sec_param)
        self.layout.addWidget(sec_act)
        self.layout.addWidget(sec_export)
        self.layout.addWidget(sec_vsi)
        self.layout.addStretch(1)

        # Bug 5: close any pre-existing bottom panel for this viewer before creating a new one
        vid = id(napari_viewer)
        if vid in CalciumControls._results_docks:
            try:
                napari_viewer.window.remove_dock_widget(CalciumControls._results_docks[vid])
            except Exception:
                pass
            del CalciumControls._results_docks[vid]

        # Create the bottom Traces & Metrics panel and attach it to the viewer
        self.results_widget = ResultsWidget(napari_viewer, controls=self)
        _dock = napari_viewer.window.add_dock_widget(
            self.results_widget, area='bottom', name='Traces & Metrics'
        )
        _dock.show()
        self._results_dock = _dock
        CalciumControls._results_docks[vid] = _dock

        # Register both panels in napari's View menu once the main window is fully built
        QTimer.singleShot(300, self._register_view_menu_actions)

    # Bug 5: clean up the bottom dock when the controls panel is closed
    def closeEvent(self, event):
        vid = id(self.viewer)
        if vid in CalciumControls._results_docks:
            try:
                self.viewer.window.remove_dock_widget(CalciumControls._results_docks[vid])
            except Exception:
                pass
            del CalciumControls._results_docks[vid]
        try:
            self.viewer.mouse_drag_callbacks.remove(self.results_widget.on_click)
        except (ValueError, AttributeError):
            pass
        super().closeEvent(event)

    # Bug 4: restore the bottom panel after user closes it; recreate if C++ object was deleted
    def _show_results_panel(self):
        if not hasattr(self, '_results_dock') or self._results_dock is None:
            self._recreate_results_dock()
            return
        try:
            self._results_dock.show()
            self._results_dock.raise_()
        except RuntimeError:
            # Dock's C++ object was deleted when the user closed it — recreate it
            self._recreate_results_dock()

    def _recreate_results_dock(self):
        vid = id(self.viewer)
        try:
            _dock = self.viewer.window.add_dock_widget(
                self.results_widget, area='bottom', name='Traces & Metrics'
            )
            _dock.show()
            self._results_dock = _dock
            CalciumControls._results_docks[vid] = _dock
        except Exception as e:
            QMessageBox.warning(self, "Panel Error", f"Could not restore Traces panel:\n{e}")

    # ------------------------------------------------------------------
    # Queue & file loading
    # ------------------------------------------------------------------

    def _on_mode_changed(self, text):
        # UI #1: keep the spin-box label in sync with the selected mode
        self.lbl_val.setText(f"{text}:")
        self._update_frame_info()

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

    # UI #2: colour-code each queue item by its processing state
    _STATE_COLORS = {
        'unprocessed': None,          # default (no tint)
        'analysed':    '#FFC107',     # amber
        'verified':    '#4CAF50',     # green
    }

    def _set_file_status(self, path, status):
        """Update the list-item background to reflect analysis / verification state."""
        self._file_states[path] = status
        color_hex = self._STATE_COLORS.get(status)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                if color_hex:
                    c = QColor(color_hex)
                    c.setAlpha(80)
                    item.setBackground(c)
                else:
                    item.setBackground(QColor(0, 0, 0, 0))
                break

    def add_files_to_queue(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", "Bio-Formats (*.tif *.tiff *.vsi)"
        )
        if files:
            for f in files:
                item = QListWidgetItem(os.path.basename(f))
                item.setData(Qt.ItemDataRole.UserRole, f)
                self.list_widget.addItem(item)
                # UI #2: new items start unprocessed (no tint)
                self._file_states.setdefault(f, 'unprocessed')
            if self.list_widget.currentRow() == -1:
                self.list_widget.setCurrentRow(0)

    def on_queue_item_changed(self, current, previous):
        if not current:
            return
        self.load_file(current.data(Qt.ItemDataRole.UserRole))

    # ------------------------------------------------------------------
    # Per-file UI state persistence
    # ------------------------------------------------------------------

    def _snapshot_ui_state(self):
        """Capture the current file's full UI state before switching away."""
        if self.last_path is None:
            return
        if not hasattr(self, 'results_widget') or self.results_widget.results is None:
            return
        self._file_ui_state[self.last_path] = {
            'results':         self.results_widget.results,
            'bin_size':        self.results_widget.bin_size,
            'selected_coords': list(self.results_widget.selected_coords),
            'lbl_beats':       self.lbl_beats.text(),
            'lbl_sync':        self.lbl_sync.text(),
        }

    def _restore_ui_state(self, fname):
        """Rebuild labels, napari layers and trace panel from a saved file state."""
        state    = self._file_ui_state[fname]
        results  = state['results']
        bin_size = state['bin_size']
        s        = bin_size
        scale    = (s, s)
        translate = ((s - 1) / 2, (s - 1) / 2)

        self.lbl_beats.setText(state['lbl_beats'])
        self.lbl_sync.setText(state['lbl_sync'])
        self.prog.setValue(100)

        self.viewer.add_image(
            results['pulsatility_map'],
            name='Pulsatility', scale=scale, translate=translate, opacity=0.5,
            colormap='inferno', blending='additive', visible=False,
        )
        act_map = results['activation_map']
        if not np.all(np.isnan(act_map)):
            self.viewer.add_image(
                np.nan_to_num(act_map, nan=0.0),
                name='Wave Map', scale=scale, translate=translate, opacity=0.6,
                colormap='twilight_shifted', blending='additive',
            )
        self.viewer.add_labels(results['clu_map'], name='Clusters',
                               scale=scale, translate=translate, opacity=0.45)
        pts = self.viewer.add_points(name='Selection', ndim=3, size=s * 2)
        pts.face_color = 'transparent'
        pts.edge_color = 'transparent'
        pts.mode       = 'pan_zoom'

        self.results_widget.results         = results
        self.results_widget.bin_size        = bin_size
        self.results_widget.selected_coords = state['selected_coords']
        self.results_widget.refresh_ui()

    def load_file(self, fname):
        try:
            # Snapshot the outgoing file's full UI state before wiping layers
            self._snapshot_ui_state()

            self.viewer.layers.clear()
            self.raw_stack = load_image(fname)

            if self.raw_stack.ndim == 2:
                self.raw_stack = self.raw_stack[np.newaxis, ...]

            self.viewer.add_image(self.raw_stack, name=os.path.basename(fname))
            self.last_path = fname
            self.btn_run.setEnabled(True)

            # Bug 10: reset analysis state for files not yet processed
            self.processed_results = None
            self.lbl_beats.setText("Beats detected: —")
            self.lbl_sync.setText("Sync index: —")
            if hasattr(self, 'results_widget'):
                self.results_widget.reset()

            # Bug 8: re-enable Verify button for the newly loaded file
            self.btn_save_next.setEnabled(True)

            T, H, W = self.raw_stack.shape
            timing = read_file_timing(fname)

            if timing['fps']:
                self.combo_mode.setCurrentText("FPS")
                self.spin_val.setValue(timing['fps'])
                self._fps_source = timing['source']
            else:
                self._fps_source = None
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
                    if fname in self._file_ui_state:
                        self._restore_ui_state(fname)
                    elif self.chk_auto.isChecked():
                        QMessageBox.warning(
                            self, "FPS Not Detected",
                            f"No FPS metadata found in:\n{os.path.basename(fname)}\n\n"
                            f"Current setting gives {est_dur:.0f}s for {T} frames, which looks wrong.\n\n"
                            f"Please set the correct FPS or Duration and click 'Run Analysis'."
                        )
                    return
                self._update_frame_info()

            self.spin_bin.setValue(16 if max(H, W) < 2048 else 32)
            self._update_frame_info()

            if fname in self._file_ui_state:
                self._restore_ui_state(fname)
            elif self.chk_auto.isChecked():
                self.start_analysis()

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Could not load file.\n\nError: {e}\n\nDid you install aicsimageio?"
            )

    def remove_selected_file(self):
        row = self.list_widget.currentRow()
        if row < 0:
            return

        # Bug 9+11: capture path before removal so we can purge all associated state
        item = self.list_widget.item(row)
        path = item.data(Qt.ItemDataRole.UserRole) if item else None

        self.list_widget.takeItem(row)

        if path:
            fname = os.path.basename(path)
            # Purge verified export data for this file
            self.master_results = [r for r in self.master_results if r.get('Filename') != fname]
            self.master_traces  = [t for t in self.master_traces  if t.get('Filename') != fname]
            # Purge analysis cache so re-adding the file triggers a fresh run
            self._results_cache = {k: v for k, v in self._results_cache.items() if k[0] != path}
            self._verified_paths.discard(path)
            self._file_states.pop(path, None)
            # UI #3+4: update counter and disable export if nothing left
            n_cells = len(self.master_results)
            n_files = len(self._verified_paths)
            self.lbl_master_count.setText(
                f"Verified Cells: {n_cells} across {n_files} file(s)" if n_cells else "Verified Cells: 0"
            )
            self.btn_export_master.setEnabled(n_cells > 0)

        if self.list_widget.count() == 0:
            self.viewer.layers.clear()
            self.raw_stack = None
            self.last_path = None
            self.btn_run.setEnabled(False)
            self.btn_save_next.setEnabled(False)
            if hasattr(self, 'results_widget'):
                self.results_widget.reset()

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
        # Bug 8: re-enable Verify after a fresh analysis run so user can save results
        self.btn_save_next.setEnabled(True)
        self.prog.setValue(100)

        if hasattr(self, '_pending_cache_key'):
            self._results_cache[self._pending_cache_key] = results
            del self._pending_cache_key

        self.lbl_beats.setText(f"Beats detected: {results['beat_count']}")
        self.lbl_sync.setText(f"Sync index: {results['sync_index']:.3f}")
        # UI #2: mark file as analysed (amber) unless already verified (green)
        if self.last_path and self._file_states.get(self.last_path) != 'verified':
            self._set_file_status(self.last_path, 'analysed')

        s = self.spin_bin.value()
        scale     = (s, s)
        # Bug 1: translate centers each bin over its corresponding raw-image region.
        # In napari's voxel-center model, bin (i,j) with scale=s is displayed at world
        # (i*s, j*s). The center of the raw pixels it covers is at ((s-1)/2 + i*s, ...).
        # Adding this translate aligns the overlays with the underlying image.
        translate = ((s - 1) / 2, (s - 1) / 2)

        for name in ['Clusters', 'Pulsatility', 'Wave Map', 'Selection']:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

        self.viewer.add_image(
            results['pulsatility_map'],
            name='Pulsatility', scale=scale, translate=translate, opacity=0.5,
            colormap='inferno', blending='additive', visible=False,
        )

        act_map = results['activation_map']
        if not np.all(np.isnan(act_map)):
            self.viewer.add_image(
                np.nan_to_num(act_map, nan=0.0),
                name='Wave Map', scale=scale, translate=translate, opacity=0.6,
                colormap='twilight_shifted', blending='additive',
            )

        self.viewer.add_labels(results['clu_map'], name='Clusters',
                               scale=scale, translate=translate, opacity=0.45)

        pts = self.viewer.add_points(name='Selection', ndim=3, size=s * 2)
        pts.face_color = 'transparent'
        pts.edge_color = 'transparent'
        pts.mode       = 'pan_zoom'

        if hasattr(self, 'results_widget'):
            self.results_widget.set_data(results, s)
            if self.chk_auto.isChecked():
                self.results_widget.random_sample()

    # ------------------------------------------------------------------
    # Export workflow
    # ------------------------------------------------------------------

    def save_and_next(self):
        if not self.last_path or not hasattr(self, 'results_widget'):
            return

        # Bug 10: refuse to save when no analysis has been run for this file
        if self.processed_results is None:
            QMessageBox.warning(
                self, "No Analysis Results",
                "No analysis has been run for this file.\n"
                "Please run analysis before saving."
            )
            return

        # Bug 8: guard against accidental duplicate saves
        if self.last_path in self._verified_paths:
            ans = QMessageBox.question(
                self, "Already Saved",
                f"{os.path.basename(self.last_path)} has already been saved to the export list.\n\n"
                "Save again? This will add duplicate rows to the export.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.No:
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
        self._verified_paths.add(self.last_path)
        # UI #2: mark file green once verified
        self._set_file_status(self.last_path, 'verified')
        # UI #3+4: update counter and enable export
        n_cells = len(self.master_results)
        n_files = len(self._verified_paths)
        self.lbl_master_count.setText(f"Verified Cells: {n_cells} across {n_files} file(s)")
        self.btn_export_master.setEnabled(n_cells > 0)

        # Disable Verify button until a new analysis is run or a different file is loaded
        self.btn_save_next.setEnabled(False)

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
            metric_col_order = [
                'Filename', 'Cell', 'X (Binned)', 'Y (Binned)',
                'BPM', 'Amp', 'F0',
                'T_ON_ms', 'T10_ON', 'T50_ON', 'T90_ON',
                'T_OFF_ms', 'T10_OFF', 'T50_OFF', 'T90_OFF',
                'CD',
            ]
            df_metrics = pd.DataFrame(self.master_results)
            present = [c for c in metric_col_order if c in df_metrics.columns]
            df_metrics = df_metrics[present].round(3)

            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                df_metrics.to_excel(writer, sheet_name='Metrics', index=False)

                if self.master_traces:
                    time_ref = self.master_traces[0]['time']
                    df_traces = pd.DataFrame({'Time (s)': time_ref})
                    for tr in self.master_traces:
                        col_label = (
                            f"{os.path.splitext(tr['Filename'])[0]}"
                            f"_P{tr['Cell'].lstrip('P')}"
                            f"_Y{tr['Y']}X{tr['X']}"
                        )
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

    def _toggle_collapse_all(self):
        """Collapse all sections if any are expanded; expand all if all are collapsed."""
        self._collapsed_all = not self._collapsed_all
        for sec in self._all_sections:
            sec.set_expanded(not self._collapsed_all)
        self._btn_collapse_all.setText(
            "⊞  Expand All" if self._collapsed_all else "⊟  Collapse All"
        )

    def _register_view_menu_actions(self):
        """Add CTA Controls and Traces & Metrics toggles to napari's View menu."""
        try:
            menubar = self.viewer.window._qt_window.menuBar()
            view_menu = None
            for action in menubar.actions():
                if 'view' in action.text().lower():
                    view_menu = action.menu()
                    break
            if view_menu is None:
                return

            view_menu.addSeparator()

            # --- Traces & Metrics toggle ---
            traces_action = QAction("Traces && Metrics", view_menu)
            traces_action.setCheckable(True)
            traces_action.setChecked(True)
            def _toggle_traces(checked):
                dock = getattr(self, '_results_dock', None)
                if dock is None:
                    return
                try:
                    dock.show() if checked else dock.hide()
                except RuntimeError:
                    pass
            traces_action.toggled.connect(_toggle_traces)
            view_menu.addAction(traces_action)
            self._traces_view_action = traces_action

            # --- CTA Controls toggle ---
            # Walk up the parent chain to find the dock wrapper that contains this widget
            ctrl_dock = None
            p = self.parent()
            for _ in range(10):
                if p is None:
                    break
                cls_name = type(p).__name__
                if cls_name in ('QtViewerDockWidget', 'QDockWidget', '_QtMainWindow'):
                    ctrl_dock = p
                    break
                try:
                    p = p.parent()
                except Exception:
                    break

            ctrl_action = QAction("CTA Controls", view_menu)
            ctrl_action.setCheckable(True)
            ctrl_action.setChecked(True)
            def _toggle_ctrl(checked, _dock=ctrl_dock):
                target = _dock if _dock is not None else self
                try:
                    target.setVisible(checked)
                except RuntimeError:
                    pass
            ctrl_action.toggled.connect(_toggle_ctrl)
            view_menu.addAction(ctrl_action)
            self._ctrl_view_action = ctrl_action

        except Exception:
            pass  # View menu integration is best-effort; never crash on it

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

        self.setMinimumHeight(180)
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
        cols = [
            "Cell",
            "BPM", "Amp", "F0",
            "T_ON_ms", "T10_ON", "T50_ON", "T90_ON",
            "T_OFF_ms", "T10_OFF", "T50_OFF", "T90_OFF",
            "CD",
        ]
        # UI #9: full-name tooltips for each abbreviated column header
        col_tips = [
            "Cell identifier",
            "Beats Per Minute — spontaneous firing rate",
            "Amplitude — peak ΔF/F above baseline",
            "Baseline fluorescence (F₀)",
            "Rise time: 10% → 90% of peak (ms)",
            "Time from 10% of rise to peak (ms)",
            "Time from 50% of rise to peak (ms)",
            "Time from 90% of rise to peak (ms)",
            "Decay time: 90% → 10% of peak (ms)",
            "Decay from peak to 10% of amplitude (ms)",
            "Decay from peak to 50% of amplitude (ms)",
            "Decay from peak to 90% of amplitude (ms)",
            "Calcium transient Duration at 50% amplitude (ms)",
        ]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        # UI #9: apply tooltips
        for i, tip in enumerate(col_tips):
            self.table.horizontalHeaderItem(i).setToolTip(tip)
        # Bug 7: stretch columns to fill available width instead of sizing to content
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.table)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([800, 1200])  # UI #10: 40/60 default split
        layout.addWidget(self.splitter)

        self.viewer.mouse_drag_callbacks.append(self.on_click)
        self.viewer.dims.events.current_step.connect(self.update_points_z)

    def set_data(self, results, bin_size):
        self.results  = results
        self.bin_size = bin_size
        self.selected_coords = []
        self.refresh_ui()

    def reset(self):
        """Clear all analysis state — called when switching to a file with no results."""
        self.results = None
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
        raw_sigs   = self.results.get('raw_signals')
        W          = self.results['dims'][1]
        beat_peaks = self.results.get('beat_peaks', np.array([]))
        for i, (y, x) in enumerate(self.selected_coords):
            idx  = y * W + x
            raw  = raw_sigs[idx] if raw_sigs is not None else None
            m    = extract_beat_averaged_features(time, sigs[idx], beat_peaks, raw_signal=raw)
            if m:
                m.pop('CD_estimated', None)
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
        active_idx = np.where(labels >= -1)[0]
        if len(active_idx) == 0:
            self.selected_coords = []
            self.refresh_ui()
            return

        sigs  = self.results['corrected_signals'][active_idx]
        amps  = np.max(sigs, axis=1) - np.min(sigs, axis=1)
        w     = amps ** 2
        probs = w / np.sum(w) if np.sum(w) > 0 else None
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
        """Generator callback — distinguishes plain clicks from pan drags.

        Napari fires mouse_drag_callbacks as generators: the code before the
        first yield runs on press, the while loop consumes move events, and
        the code after the loop runs on release.  We only add/remove a point
        when the mouse did not move between press and release (i.e. it was a
        click, not a pan drag).
        """
        # Bug 2+3: early exits before yielding mean napari treats this as a
        # no-op generator and does not interfere with the pan/zoom tool.
        if getattr(event, 'button', 0) != 1:
            return
        if self.results is None:
            return
        # Bug 3: respect the Selection layer's visibility — hidden means inactive
        if 'Selection' not in self.viewer.layers or not self.viewer.layers['Selection'].visible:
            return

        # Yield once to receive subsequent move/release events
        dragged = False
        yield
        while event.type == 'mouse_move':
            dragged = True
            yield

        # Bug 2: only act on a pure click (no drag movement)
        if dragged:
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

        # UI #7: show current filename as the plot title
        fname = getattr(self.controls, 'last_path', None)
        if fname:
            self.ax.set_title(os.path.basename(fname), fontsize=9, color='#90CAF9', pad=3)

        if not self.results or not self.selected_coords:
            # UI #6: placeholder text when no cells are selected
            self.ax.text(
                0.5, 0.5,
                "Click on the image to select cells",
                transform=self.ax.transAxes,
                ha='center', va='center',
                fontsize=11, style='italic',
                color='#888888',
            )
            self.canvas.draw()
            return

        time     = self.results['time']
        sigs     = self.results['corrected_signals']
        raw_sigs = self.results.get('raw_signals')
        W        = self.results['dims'][1]
        t_idx = self.viewer.dims.current_step[0] if len(self.viewer.dims.current_step) > 2 else 0

        points_data = []
        face_colors = []
        text_labels = []

        beat_peaks = self.results.get('beat_peaks', np.array([]))

        for i, (y, x) in enumerate(self.selected_coords):
            idx   = y * W + x
            sig   = sigs[idx]
            raw   = raw_sigs[idx] if raw_sigs is not None else None
            color = self.colors[i % len(self.colors)]

            self.ax.plot(time, sig, color=color, label=f"P{i+1}", linewidth=1.2)

            if len(beat_peaks) > 0:
                self.ax.plot(time[beat_peaks], sig[beat_peaks], 'v',
                             color=color, markersize=5, alpha=0.7)

            m = extract_beat_averaged_features(time, sig, beat_peaks, raw_signal=raw)
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

            # UI #8: tint every cell in this row with a 12%-alpha version of the trace colour
            row_bg = QColor(color)
            row_bg.setAlpha(30)
            for c_idx in range(self.table.columnCount()):
                it = self.table.item(r, c_idx)
                if it:
                    it.setBackground(row_bg)

            # Bug 1: use (bin_size-1)/2 offset so the dot sits at the visual center
            # of the bin as displayed by the translated overlay layers.
            offset = (self.bin_size - 1) / 2
            py = offset + y * self.bin_size
            px = offset + x * self.bin_size
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
    ctrl   = CalciumControls(viewer)
    viewer.window.add_dock_widget(ctrl, area='right', name='CTA Controls')
    napari.run()


if __name__ == '__main__':
    main()
