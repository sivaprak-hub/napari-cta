import os
import numpy as np
import pandas as pd
import tifffile
import napari

import matplotlib
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QSpinBox, QDoubleSpinBox, QComboBox,
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

from scipy.signal import find_peaks as _scipy_find_peaks
from .backend import (AnalysisWorker, BatchWorker,
                       extract_detailed_features, extract_beat_averaged_features,
                       load_image, convert_single_vsi, read_file_timing,
                       save_fps_sidecar)

# ─────────────────────────────────────────────────────────────────────────────
# Theme stylesheet — object-name rules (QLabel#badge_green etc.) mean a
# single setStyleSheet() call on the top-level widget updates every child.
# ─────────────────────────────────────────────────────────────────────────────
_PANEL_STYLE_DARK = """
QWidget { background: #1e2130; color: #dce1f0; font-size: 12px; }
QScrollArea, QScrollArea > QWidget > QWidget { background: #1e2130; border: none; }
QScrollBar:vertical {
    background: #181926; width: 6px; border-radius: 3px; margin: 0;
}
QScrollBar::handle:vertical { background: #404870; border-radius: 3px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QPushButton {
    background: #272b42; color: #c8d0ec; border: 1px solid #3a3e58;
    border-radius: 5px; padding: 5px 10px; min-height: 24px; font-size: 11px;
}
QPushButton:hover  { background: #343857; border-color: #5a6088; color: #e8ecff; }
QPushButton:pressed  { background: #1d2035; }
QPushButton:disabled { color: #3e4260; background: #1e2030; border-color: #282c40; }

QComboBox {
    background: #232640; border: 1px solid #3a3e58; border-radius: 4px;
    padding: 3px 8px; color: #c8d0ec; min-height: 22px;
}
QComboBox:hover { border-color: #5a6088; }
QComboBox::drop-down { border: none; width: 16px; }
QComboBox QAbstractItemView {
    background: #232640; border: 1px solid #3a3e58;
    selection-background-color: #3d4570; color: #c8d0ec; outline: none;
}

QSpinBox, QDoubleSpinBox {
    background: #232640; border: 1px solid #3a3e58; border-radius: 4px;
    padding: 3px 24px 3px 6px; color: #c8d0ec; min-height: 22px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #5a6088; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 18px; background: #2a2e48;
    border-left: 1px solid #3a3e58; border-bottom: 1px solid #3a3e58;
    border-top-right-radius: 4px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 18px; background: #2a2e48;
    border-left: 1px solid #3a3e58;
    border-bottom-right-radius: 4px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #3a3e60; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { width: 7px; height: 7px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { width: 7px; height: 7px; }

QListWidget {
    background: #171a2c; border: 1px solid #3a3e58; border-radius: 4px; padding: 2px;
}
QListWidget::item          { padding: 3px 6px; border-radius: 3px; }
QListWidget::item:selected { background: #3d4570; color: #ffffff; }
QListWidget::item:hover:!selected { background: #272b42; }

QProgressBar {
    background: #171a2c; border: 1px solid #3a3e58; border-radius: 5px;
    max-height: 8px; color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #5c6bc0, stop:1 #42a5f5);
    border-radius: 5px;
}

QCheckBox { color: #8890b0; spacing: 6px; font-size: 11px; }
QCheckBox:hover { color: #c8d0ec; }
QCheckBox::indicator {
    width: 13px; height: 13px; border: 1px solid #3a3e58; border-radius: 3px; background: #232640;
}
QCheckBox::indicator:checked { background: #5c7cfa; border-color: #5c7cfa; }

QTableWidget {
    background: #171a2c; alternate-background-color: #1d2038; color: #c8d0ec;
    border: 1px solid #3a3e58; border-radius: 4px; gridline-color: #282c45;
    selection-background-color: #3d4570; selection-color: #ffffff;
}
QHeaderView::section {
    background: #232640; color: #90caf9; border: none;
    border-right: 1px solid #3a3e58; border-bottom: 2px solid #5c7cfa;
    font-weight: 600; font-size: 10px; padding: 4px 3px;
}
QHeaderView::section:hover { background: #2d3255; }

QSplitter::handle            { background: #3a3e58; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical   { height: 2px; }
QLabel { background: transparent; }

QSlider::groove:horizontal { height: 4px; border-radius: 2px; background: #2e3248; }
QSlider::handle:horizontal {
    width: 10px; height: 10px; margin: -3px 0; border-radius: 5px;
    background: #5c7cfa; border: none;
}
QSlider::sub-page:horizontal { background: #5c7cfa; border-radius: 2px; }

QLabel#badge_green {
    background: #1a2e1a; border: 1px solid #2e5c2e; border-radius: 8px;
    padding: 2px 8px; color: #69db7c; font-size: 11px; font-weight: 600;
}
QLabel#badge_blue {
    background: #1a1e3a; border: 1px solid #2e3a6e; border-radius: 8px;
    padding: 2px 8px; color: #74c0fc; font-size: 11px;
}
QPushButton#btn_danger {
    background: #2d1a1a; border: 1px solid #5c2020; color: #e57373;
    border-radius: 5px; padding: 5px 8px;
}
QPushButton#btn_danger:hover { background: #3d2020; border-color: #8c3030; color: #ff8a80; }
QPushButton#btn_danger:pressed { background: #1d0e0e; }
QPushButton#btn_ghost {
    font-size: 11px; color: #6870a0; border: 1px dashed #333650;
    background: #1a1d2e; border-radius: 4px;
}
QPushButton#btn_ghost:hover { color: #9090c0; border-color: #5060a0; background: #1e2138; }
"""

_PANEL_STYLE = _PANEL_STYLE_DARK  # default theme alias

_PANEL_STYLE_LIGHT = """
QWidget { background: #f5f6fa; color: #1e2340; font-size: 12px; }
QScrollArea, QScrollArea > QWidget > QWidget { background: #f5f6fa; border: none; }
QScrollBar:vertical {
    background: #dde3f5; width: 6px; border-radius: 3px; margin: 0;
}
QScrollBar::handle:vertical { background: #a0aed0; border-radius: 3px; min-height: 20px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QPushButton {
    background: #e8ecf8; color: #2a3560; border: 1px solid #c0cce8;
    border-radius: 5px; padding: 5px 10px; min-height: 24px; font-size: 11px;
}
QPushButton:hover  { background: #d8e0f5; border-color: #7a8cc0; color: #1a2040; }
QPushButton:pressed  { background: #c8d4f0; }
QPushButton:disabled { color: #a0aec0; background: #edf0f8; border-color: #c8d4e8; }

QComboBox {
    background: #ffffff; border: 1px solid #c0cce8; border-radius: 4px;
    padding: 3px 8px; color: #2a3560; min-height: 22px;
}
QComboBox:hover { border-color: #7a8cc0; }
QComboBox::drop-down { border: none; width: 16px; }
QComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #c0cce8;
    selection-background-color: #d0daf8; color: #2a3560; outline: none;
}

QSpinBox, QDoubleSpinBox {
    background: #ffffff; border: 1px solid #c0cce8; border-radius: 4px;
    padding: 3px 24px 3px 6px; color: #2a3560; min-height: 22px;
}
QSpinBox:hover, QDoubleSpinBox:hover { border-color: #7a8cc0; }
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; subcontrol-position: top right;
    width: 18px; background: #e8ecf8;
    border-left: 1px solid #c0cce8; border-bottom: 1px solid #c0cce8;
    border-top-right-radius: 4px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; subcontrol-position: bottom right;
    width: 18px; background: #e8ecf8;
    border-left: 1px solid #c0cce8;
    border-bottom-right-radius: 4px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background: #d8e0f5; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { width: 7px; height: 7px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { width: 7px; height: 7px; }

QListWidget {
    background: #ffffff; border: 1px solid #c0cce8; border-radius: 4px; padding: 2px;
}
QListWidget::item          { padding: 3px 6px; border-radius: 3px; }
QListWidget::item:selected { background: #d0daf8; color: #1a2040; }
QListWidget::item:hover:!selected { background: #e8ecf8; }

QProgressBar {
    background: #dde3f5; border: 1px solid #c0cce8; border-radius: 5px;
    max-height: 8px; color: transparent;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #5c6bc0, stop:1 #42a5f5);
    border-radius: 5px;
}

QCheckBox { color: #5060a0; spacing: 6px; font-size: 11px; }
QCheckBox:hover { color: #1e2340; }
QCheckBox::indicator {
    width: 13px; height: 13px; border: 1px solid #c0cce8; border-radius: 3px; background: #ffffff;
}
QCheckBox::indicator:checked { background: #5c7cfa; border-color: #5c7cfa; }

QTableWidget {
    background: #ffffff; alternate-background-color: #f0f4fc; color: #2a3560;
    border: 1px solid #c0cce8; border-radius: 4px; gridline-color: #dde5f5;
    selection-background-color: #d0daf8; selection-color: #1a2040;
}
QHeaderView::section {
    background: #e8ecf8; color: #3050a0; border: none;
    border-right: 1px solid #c0cce8; border-bottom: 2px solid #5c7cfa;
    font-weight: 600; font-size: 10px; padding: 4px 3px;
}
QHeaderView::section:hover { background: #d8e0f5; }

QSplitter::handle            { background: #c0cce8; }
QSplitter::handle:horizontal { width: 2px; }
QSplitter::handle:vertical   { height: 2px; }
QLabel { background: transparent; }

QSlider::groove:horizontal { height: 4px; border-radius: 2px; background: #c8d4e8; }
QSlider::handle:horizontal {
    width: 10px; height: 10px; margin: -3px 0; border-radius: 5px;
    background: #5c7cfa; border: none;
}
QSlider::sub-page:horizontal { background: #5c7cfa; border-radius: 2px; }

QLabel#badge_green {
    background: #e8f8ee; border: 1px solid #90d0a8; border-radius: 8px;
    padding: 2px 8px; color: #1a6e3a; font-size: 11px; font-weight: 600;
}
QLabel#badge_blue {
    background: #e8f0ff; border: 1px solid #90aef0; border-radius: 8px;
    padding: 2px 8px; color: #2050c0; font-size: 11px;
}
QPushButton#btn_danger {
    background: #fde8e8; border: 1px solid #e0a0a0; color: #b03030;
    border-radius: 5px; padding: 5px 8px;
}
QPushButton#btn_danger:hover { background: #fcd0d0; border-color: #c06060; color: #8c2020; }
QPushButton#btn_danger:pressed { background: #f8c0c0; }
QPushButton#btn_ghost {
    font-size: 11px; color: #6070a0; border: 1px dashed #a0b0d0;
    background: #f0f4fc; border-radius: 4px;
}
QPushButton#btn_ghost:hover { color: #3050a0; border-color: #5070c0; background: #e8ecf8; }
"""




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

    # Cycle through accent colours to give each section a distinct identity
    _ACCENT_COLORS = ['#5c7cfa', '#20c997', '#f59f00', '#74c0fc', '#e64980']
    _instance_count = 0

    def __init__(self, title, parent=None, expanded=True):
        super().__init__(parent)
        # Pick a unique accent colour for this section instance
        self._accent = CollapsibleSection._ACCENT_COLORS[
            CollapsibleSection._instance_count % len(CollapsibleSection._ACCENT_COLORS)
        ]
        CollapsibleSection._instance_count += 1

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 4)
        root.setSpacing(0)

        self._title = title

        self._btn = QPushButton()
        self._btn.setCheckable(True)
        self._btn.setChecked(expanded)
        self._btn.toggled.connect(self._on_toggled)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 6, 8, 8)
        self._body_layout.setSpacing(5)
        self._body.setVisible(expanded)

        root.addWidget(self._btn)
        root.addWidget(self._body)
        self._refresh_label(expanded)
        self.apply_theme('dark')  # default; updated by _apply_theme() on parent

    def apply_theme(self, theme: str = 'dark'):
        """Reapply button + body stylesheets, preserving accent colour."""
        if theme == 'light':
            btn_bg    = 'qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #e8ecf8,stop:1 #dde3f5)'
            btn_hover = 'qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #d8e0f5,stop:1 #ccd4f0)'
            btn_chk   = 'qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #d0d8f5,stop:1 #dde3f5)'
            btn_top   = '#c0cce8'
            btn_color = '#1e2340'
            body_bg   = '#f0f4fc'
            body_bdr  = '#c8d4e8'
        else:
            btn_bg    = 'qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #252840,stop:1 #1e2130)'
            btn_hover = 'qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #2d3255,stop:1 #252840)'
            btn_chk   = 'qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #2d3255,stop:1 #1e2130)'
            btn_top   = '#2e3248'
            btn_color = '#c5cae9'
            body_bg   = '#191c2e'
            body_bdr  = '#2e3248'
        self._btn.setStyleSheet(
            f"QPushButton {{"
            f"  text-align: left; font-weight: 600; font-size: 11px; padding: 6px 8px;"
            f"  background: {btn_bg}; border: none;"
            f"  border-left: 3px solid {self._accent}; border-top: 1px solid {btn_top};"
            f"  border-radius: 0px; color: {btn_color}; letter-spacing: 0.3px;"
            f"}}"
            f"QPushButton:hover {{ background: {btn_hover}; }}"
            f"QPushButton:checked {{ background: {btn_chk}; }}"
        )
        self._body.setStyleSheet(
            f"QWidget {{ background: {body_bg};"
            f"  border-left: 1px solid {body_bdr}; border-right: 1px solid {body_bdr};"
            f"  border-bottom: 1px solid {body_bdr}; }}"
        )

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


class GraphWindow(QWidget):
    """Floating graph window that hosts the matplotlib canvas independently of the bottom panel."""

    def __init__(self, results_widget, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self._rw = results_widget
        self.setWindowTitle("Calcium Transient Traces")
        self.setMinimumSize(500, 320)
        self._theme = 'dark'

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(0)

    def auto_resize(self, n_traces: int):
        """Resize window to fit n_traces traces comfortably."""
        w = max(620, min(1600, 480 + n_traces * 110))
        h = max(380, min(900, 360 + n_traces * 12))
        self.resize(w, h)

    def apply_theme(self, theme: str = 'dark'):
        self._theme = theme
        bg = '#f5f6fa' if theme == 'light' else '#171a2c'
        self.setStyleSheet(f"QWidget {{ background: {bg}; }}")

    def closeEvent(self, event):
        event.ignore()
        rw = self._rw
        if rw is not None and hasattr(rw, '_content_splitter'):
            rw._content_splitter.insertWidget(0, rw.canvas)
            rw.canvas.show()
            w = rw._content_splitter.width()
            if w > 20:
                rw._content_splitter.setSizes([w // 2, w // 2])
        self.hide()


class CalciumControls(QWidget):
    """Left-panel controls — napari plugin widget."""

    # Bug 5: class-level registry prevents duplicate bottom panels across re-opens.
    _results_docks = {}   # id(viewer) → results dock widget

    def __init__(self, napari_viewer: 'napari.viewer.Viewer'):
        super().__init__()
        self.viewer            = napari_viewer
        self._theme            = getattr(napari_viewer, 'theme', 'dark')
        self.setStyleSheet(_PANEL_STYLE_LIGHT if self._theme == 'light' else _PANEL_STYLE_DARK)
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

        # ── Plugin title header ─────────────────────────────────────────────
        self._header_widget = QWidget()
        h_lay = QVBoxLayout(self._header_widget)
        h_lay.setContentsMargins(8, 8, 8, 6)
        h_lay.setSpacing(2)

        # Title row: icon+title
        title_row = QHBoxLayout()
        title_row.setSpacing(6)

        self.lbl_title = QLabel("Calcium Transient Analyzer")
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_row.addStretch(1)
        title_row.addWidget(self.lbl_title, 5)
        title_row.addStretch(1)

        self.lbl_current_file = QLabel("No file loaded")
        self.lbl_current_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_current_file.setWordWrap(True)

        h_lay.addLayout(title_row)
        h_lay.addWidget(self.lbl_current_file)
        outer.addWidget(self._header_widget)
        self._apply_header_theme(self._theme)

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
        self._btn_collapse_all = QPushButton("Collapse All")
        self._btn_collapse_all.setStyleSheet(
            "font-size: 10px; padding: 3px 8px;"
            "background: #1e2130; border: 1px solid #333650;"
            "border-radius: 3px; color: #6870a0;"
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
        btn_remove     = QPushButton("Remove")
        btn_remove.setObjectName("btn_danger")
        btn_remove.clicked.connect(self.remove_selected_file)
        btn_remove_all = QPushButton("Clear All")
        btn_remove_all.setObjectName("btn_danger")
        btn_remove_all.clicked.connect(self.remove_all_files)
        btn_layout.addWidget(btn_add_files)
        btn_layout.addWidget(btn_remove)
        btn_layout.addWidget(btn_remove_all)

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
        self.btn_run = QPushButton("▶  Run Analysis")
        self.btn_run.setStyleSheet(
            "QPushButton {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #4a72e0, stop:1 #3558b8);"
            "  color: white; font-weight: bold; font-size: 12px;"
            "  border: 1px solid #3050a0; border-radius: 5px; padding: 7px;"
            "  min-height: 28px;"
            "}"
            "QPushButton:hover {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #5a84f4, stop:1 #4468cc);"
            "}"
            "QPushButton:pressed  { background: #2a46a0; }"
            "QPushButton:disabled { background: #1e2130; color: #3e4260; border-color: #282c40; }"
        )
        self.btn_run.clicked.connect(self.start_analysis)
        self.btn_run.setEnabled(False)
        self.prog = QProgressBar()
        self.lbl_beats = QLabel("Beats detected: —")
        self.lbl_beats.setObjectName("badge_green")
        self.lbl_sync  = QLabel("Sync index: —")
        self.lbl_sync.setObjectName("badge_blue")
        # Bug 4: restore bottom metrics panel; separate button opens floating graph window
        self.btn_show_traces = QPushButton("Show Metrics Panel")
        self.btn_show_traces.setObjectName("btn_ghost")
        self.btn_show_traces.clicked.connect(self._show_results_panel)
        self.btn_show_graph = QPushButton("Show Graph")
        self.btn_show_graph.setObjectName("btn_ghost")
        self.btn_show_graph.clicked.connect(self._show_graph_window)
        sec_act.body_layout.addWidget(self.btn_run)
        sec_act.body_layout.addWidget(self.prog)
        sec_act.body_layout.addWidget(self.lbl_beats)
        sec_act.body_layout.addWidget(self.lbl_sync)
        sec_act.body_layout.addWidget(self.btn_show_traces)
        sec_act.body_layout.addWidget(self.btn_show_graph)

        # --- 4. GUIDED EXPORT ---
        sec_export = CollapsibleSection("4. Guided Export")
        self.btn_save_next = QPushButton("Verify")
        self.btn_save_next.setStyleSheet(
            "QPushButton {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #2a9d5c, stop:1 #1f7a45);"
            "  color: white; font-weight: bold; font-size: 12px;"
            "  border: 1px solid #1a6038; border-radius: 5px; padding: 7px;"
            "  min-height: 28px;"
            "}"
            "QPushButton:hover {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #38b36e, stop:1 #2a9d5c);"
            "}"
            "QPushButton:pressed  { background: #155d30; }"
            "QPushButton:disabled { background: #1e2130; color: #3e4260; border-color: #282c40; }"
        )
        self.btn_save_next.clicked.connect(self.save_and_next)
        self.btn_save_next.setEnabled(False)
        self.lbl_master_count = QLabel("Verified Cells: 0")
        self.lbl_master_count.setObjectName("badge_green")
        self.btn_export_master = QPushButton("Export to Excel")
        self.btn_export_master.setStyleSheet(
            "QPushButton {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #0ca678, stop:1 #087f5b);"
            "  color: white; font-weight: bold; font-size: 12px;"
            "  border: 1px solid #075e44; border-radius: 5px; padding: 7px;"
            "  min-height: 28px;"
            "}"
            "QPushButton:hover {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #12c78e, stop:1 #0ca678);"
            "}"
            "QPushButton:pressed  { background: #056040; }"
            "QPushButton:disabled { background: #1e2130; color: #3e4260; border-color: #282c40; }"
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

        # --- 6. LAYER CONTROLS ---
        sec_layers = CollapsibleSection("6. Layer Controls")
        preset_row = QHBoxLayout()
        preset_row.setSpacing(4)
        for label, slot in [("All On", self._layers_all_on),
                             ("Overlays", self._layers_overlays_only),
                             ("Raw Only", self._layers_raw_only)]:
            b = QPushButton(label)
            b.setStyleSheet("font-size: 10px; padding: 2px 5px; min-height: 20px;")
            b.clicked.connect(slot)
            preset_row.addWidget(b)
        sec_layers.body_layout.addLayout(preset_row)

        self._layer_ctrl_container = QWidget()
        self._layer_ctrl_layout    = QVBoxLayout(self._layer_ctrl_container)
        self._layer_ctrl_layout.setContentsMargins(0, 4, 0, 0)
        self._layer_ctrl_layout.setSpacing(2)
        lbl_placeholder = QLabel("Run analysis to populate layers")
        lbl_placeholder.setStyleSheet("font-style: italic; font-size: 11px;")
        self._layer_ctrl_layout.addWidget(lbl_placeholder)
        sec_layers.body_layout.addWidget(self._layer_ctrl_container)

        self._all_sections = [sec_queue, sec_param, sec_act, sec_export]
        self._all_sections_all = [sec_queue, sec_param, sec_act, sec_export, sec_layers]

        self.layout.addWidget(sec_queue)
        self.layout.addWidget(sec_param)
        self.layout.addWidget(sec_act)
        self.layout.addWidget(sec_export)
        self.layout.addWidget(sec_layers)
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

        # Connect CTA dock visibility → Traces & Metrics dock once the dock wrapper exists
        QTimer.singleShot(400, self._connect_dock_signals)
        # Apply initial theme to all panels after they render
        QTimer.singleShot(500, lambda: self._apply_theme(self._theme))
        try:
            self.viewer.events.theme.connect(self._on_napari_theme_changed)
        except Exception:
            pass

    def closeEvent(self, event):
        # Remove the Traces & Metrics bottom dock
        vid = id(self.viewer)
        if vid in CalciumControls._results_docks:
            try:
                self.viewer.window.remove_dock_widget(CalciumControls._results_docks[vid])
            except Exception:
                pass
            del CalciumControls._results_docks[vid]

        # Unregister mouse callback
        try:
            self.viewer.mouse_drag_callbacks.remove(self.results_widget.on_click)
        except (ValueError, AttributeError, RuntimeError):
            pass

        # Close the detached graph window
        try:
            gw = getattr(self.results_widget, '_graph_window', None)
            if gw is not None:
                gw.close()
        except Exception:
            pass

        # Clear all CTA layers so napari returns to its default empty state
        try:
            self.viewer.layers.clear()
        except Exception:
            pass

        # Reset napari's native layer list and layer controls to their default styling
        try:
            qt_viewer = self.viewer.window._qt_viewer
            for attr in ('dockLayerList', 'dockLayerControls'):
                dock = getattr(qt_viewer, attr, None)
                if dock is not None:
                    dock.setStyleSheet("")
        except Exception:
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

    def _show_graph_window(self):
        if hasattr(self, 'results_widget'):
            self.results_widget._show_graph_popup()

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

        # restore processed_results so "Verify, Save & Go Next" is available again
        self.processed_results = results
        self.btn_save_next.setEnabled(True)
        # Bug 4 fix: keep spin_bin in sync with the analysis that produced the current overlays
        self.spin_bin.setValue(bin_size)
        self.results_widget.results         = results
        self.results_widget.bin_size        = bin_size
        self.results_widget.selected_coords = state['selected_coords']
        self.results_widget.refresh_ui()
        QTimer.singleShot(50, self._rebuild_layer_controls)

    def load_file(self, fname):
        try:
            # Snapshot the outgoing file's full UI state before wiping layers
            self._snapshot_ui_state()

            self.viewer.layers.clear()
            self._rebuild_layer_controls()
            self.raw_stack = load_image(fname)

            if self.raw_stack.ndim == 2:
                self.raw_stack = self.raw_stack[np.newaxis, ...]

            self.viewer.add_image(self.raw_stack, name=os.path.basename(fname))
            self.last_path = fname
            self.lbl_current_file.setText(os.path.basename(fname))
            self.btn_run.setEnabled(True)

            # Bug 10: reset analysis state for files not yet processed
            self.processed_results = None
            self.lbl_beats.setText("Beats detected: —")
            self.lbl_sync.setText("Sync index: —")
            if hasattr(self, 'results_widget'):
                self.results_widget.reset()

            # Verify only makes sense after analysis — re-enabled in on_analysis_done
            self.btn_save_next.setEnabled(False)

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
                        f"No FPS in file metadata — current setting gives "
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
            # Bug 3 fix: also purge cached UI state so re-adding the file starts fresh
            self._file_ui_state.pop(path, None)
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

    def remove_all_files(self):
        if self.list_widget.count() == 0:
            return
        if QMessageBox.question(
            self, "Clear All Files",
            "Remove all files from the queue and clear all cached results?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        self.list_widget.clear()
        self._results_cache.clear()
        self._verified_paths.clear()
        self._file_states.clear()
        self._file_ui_state.clear()
        self.master_results.clear()
        self.master_traces.clear()
        self.raw_stack = None
        self.last_path = None
        self.btn_run.setEnabled(False)
        self.btn_save_next.setEnabled(False)
        self.lbl_master_count.setText("Verified Cells: 0")
        self.btn_export_master.setEnabled(False)
        self.viewer.layers.clear()
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

        _t = results['time']
        _dur = (_t[-1] - _t[0]) if len(_t) > 1 else 0
        _bpm = (results['beat_count'] / _dur * 60) if _dur > 0 else 0.0
        self.lbl_beats.setText(
            f"Beats detected: {results['beat_count']}  (~{_bpm:.0f} BPM)"
        )
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
            QTimer.singleShot(50, self._rebuild_layer_controls)

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
                    # Bug 2 fix: different files may have different frame counts / FPS.
                    # Group traces by file so each file's time column is accurate.
                    # If all files share the same time axis (same length), use a
                    # single compact format; otherwise write per-file time blocks.
                    from collections import defaultdict
                    file_groups = defaultdict(list)
                    for tr in self.master_traces:
                        file_groups[tr['Filename']].append(tr)

                    all_same_length = len({len(tr['time']) for tr in self.master_traces}) == 1

                    if all_same_length:
                        time_ref  = self.master_traces[0]['time']
                        df_traces = pd.DataFrame({'Time (s)': time_ref})
                        for tr in self.master_traces:
                            col_label = (
                                f"{os.path.splitext(tr['Filename'])[0]}"
                                f"_P{tr['Cell'].lstrip('P')}"
                                f"_Y{tr['Y']}X{tr['X']}"
                            )
                            df_traces[col_label] = tr['signal']
                    else:
                        # Different lengths — write each file's time+signals as
                        # adjacent column blocks; pad shorter blocks with NaN.
                        max_len = max(len(tr['time']) for tr in self.master_traces)
                        df_traces = pd.DataFrame()
                        for fname, trs in file_groups.items():
                            stub = os.path.splitext(fname)[0]
                            time_col  = list(trs[0]['time']) + [np.nan] * (max_len - len(trs[0]['time']))
                            df_traces[f"{stub}_Time(s)"] = time_col
                            for tr in trs:
                                col_label = (
                                    f"{stub}_P{tr['Cell'].lstrip('P')}"
                                    f"_Y{tr['Y']}X{tr['X']}"
                                )
                                sig = list(tr['signal']) + [np.nan] * (max_len - len(tr['signal']))
                                df_traces[col_label] = sig
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
        """Collapse all sections if any are expanded; expand all if all are collapsed.
        Reads actual section state rather than a flag so manual expand/collapse
        by the user never desynchronises the button label."""
        any_open = any(s.is_expanded() for s in self._all_sections)
        for sec in self._all_sections:
            sec.set_expanded(not any_open)
        self._btn_collapse_all.setText(
            "Expand All" if any_open else "Collapse All"
        )

    # ------------------------------------------------------------------
    # Dark / Light theme
    # ------------------------------------------------------------------

    def _connect_dock_signals(self):
        """Find the dock wrapping CTA Controls and sync Traces & Metrics visibility with it."""
        p = self.parent()
        for _ in range(10):
            if p is None:
                break
            cls_name = type(p).__name__
            if cls_name in ('QtViewerDockWidget', 'QDockWidget'):
                try:
                    p.visibilityChanged.connect(self._on_cta_dock_visibility)
                except Exception:
                    pass
                return
            try:
                p = p.parent()
            except Exception:
                break

    def _on_cta_dock_visibility(self, visible):
        """When CTA Controls dock is hidden/shown, sync the Traces & Metrics dock."""
        dock = getattr(self, '_results_dock', None)
        if dock is None:
            return
        try:
            if visible:
                dock.show()
            else:
                dock.hide()
        except RuntimeError:
            pass

    def _on_napari_theme_changed(self, event=None):
        theme = getattr(self.viewer, 'theme', 'dark')
        self._theme = theme
        self._apply_theme(theme)

    def _apply_header_theme(self, theme: str = 'dark'):
        """Update only the header widget's stylesheet."""
        if theme == 'light':
            bg  = 'qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #e8ecf8,stop:1 #dde3f5)'
            bdr = '#5c7cfa'
            tc  = '#1e2340'
            sc  = '#5060a0'
        else:
            bg  = 'qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #252840,stop:1 #1e2130)'
            bdr = '#5c7cfa'
            tc  = '#90caf9'
            sc  = '#5a6488'
        self._header_widget.setStyleSheet(
            f"QWidget {{ background: {bg}; border-bottom: 2px solid {bdr}; }}"
        )
        self.lbl_title.setStyleSheet(
            f"font-size: 13px; font-weight: 700; color: {tc}; "
            "letter-spacing: 0.5px; background: transparent; border: none;"
        )
        self.lbl_current_file.setStyleSheet(
            f"font-size: 10px; color: {sc}; background: transparent; border: none;"
        )

    def _apply_theme(self, theme: str = 'dark'):
        self._theme = theme
        self.setStyleSheet(_PANEL_STYLE_LIGHT if theme == 'light' else _PANEL_STYLE_DARK)
        self._apply_header_theme(theme)
        for sec in self._all_sections_all:
            sec.apply_theme(theme)
        if hasattr(self, 'results_widget'):
            self.results_widget.apply_theme(theme)
        self._rebuild_layer_controls()

    # ------------------------------------------------------------------
    # Layer Controls (section 6)
    # ------------------------------------------------------------------

    def _rebuild_layer_controls(self):
        while self._layer_ctrl_layout.count():
            item = self._layer_ctrl_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        KNOWN = ('Pulsatility', 'Wave Map', 'Clusters', 'Selection')
        present = [n for n in KNOWN if n in self.viewer.layers]

        if not present:
            lbl = QLabel("Run analysis to populate layers")
            lbl.setStyleSheet("font-style: italic; font-size: 11px;")
            self._layer_ctrl_layout.addWidget(lbl)
            return

        from qtpy.QtWidgets import QSlider
        for name in present:
            layer = self.viewer.layers[name]

            row = QWidget()
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 1, 0, 1)
            row_lay.setSpacing(6)

            chk = QCheckBox(name)
            chk.setChecked(layer.visible)
            chk.toggled.connect(lambda v, n=name: self._set_layer_visible(n, v))
            row_lay.addWidget(chk, 1)

            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(int(layer.opacity * 100))
            slider.setFixedWidth(70)
            slider.setToolTip("Opacity")
            slider.valueChanged.connect(
                lambda v, n=name: self._set_layer_opacity(n, v / 100)
            )
            row_lay.addWidget(slider)

            self._layer_ctrl_layout.addWidget(row)

    def _set_layer_visible(self, name: str, visible: bool):
        if name in self.viewer.layers:
            self.viewer.layers[name].visible = visible

    def _set_layer_opacity(self, name: str, value: float):
        if name in self.viewer.layers:
            self.viewer.layers[name].opacity = value

    def _layers_all_on(self):
        for layer in self.viewer.layers:
            layer.visible = True
        self._rebuild_layer_controls()

    def _layers_overlays_only(self):
        raw_name = os.path.basename(self.last_path) if self.last_path else None
        for layer in self.viewer.layers:
            layer.visible = layer.name != raw_name
        self._rebuild_layer_controls()

    def _layers_raw_only(self):
        raw_name = os.path.basename(self.last_path) if self.last_path else None
        for layer in self.viewer.layers:
            layer.visible = layer.name == raw_name
        self._rebuild_layer_controls()

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
        self.setStyleSheet(_PANEL_STYLE)
        self._theme = 'dark'

        import matplotlib.pyplot as plt
        cmap        = plt.get_cmap('tab20')
        self.colors = [matplotlib.colors.to_hex(cmap(i)) for i in np.linspace(0, 1, 50)]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)
        lbl_max = QLabel("Max Points:")
        lbl_max.setStyleSheet("font-size: 11px; color: #8890b0;")
        ctrl.addWidget(lbl_max)
        self.spin_max   = QSpinBox(); self.spin_max.setRange(1, 50); self.spin_max.setValue(6)
        self.btn_random = QPushButton("Random Sample")
        self.btn_random.setStyleSheet(
            "QPushButton { background: #252840; border: 1px solid #3a3e58; color: #c8d0ec;"
            "  border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #2d3255; border-color: #5c7cfa; color: #e8ecff; }"
        )
        self.btn_random.clicked.connect(self.random_sample)
        self.btn_clear  = QPushButton("Clear")
        self.btn_clear.setStyleSheet(
            "QPushButton { background: #2d1a1a; border: 1px solid #5c2020; color: #e57373;"
            "  border-radius: 4px; padding: 4px 10px; font-size: 11px; }"
            "QPushButton:hover { background: #3d2020; border-color: #8c3030; color: #ff8a80; }"
        )
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_save   = QPushButton("Save Graph")
        self.btn_save.setStyleSheet(
            "QPushButton {"
            "  background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "    stop:0 #4a72e0, stop:1 #3558b8);"
            "  color: white; font-weight: bold; font-size: 11px;"
            "  border: 1px solid #3050a0; border-radius: 4px; padding: 4px 12px;"
            "}"
            "QPushButton:hover { background: qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            "  stop:0 #5a84f4, stop:1 #4468cc); }"
        )
        self.btn_save.clicked.connect(self.save_graph)
        ctrl.addWidget(self.spin_max)
        ctrl.addWidget(self.btn_random)
        ctrl.addWidget(self.btn_clear)
        ctrl.addStretch()
        ctrl.addWidget(self.btn_save)
        layout.addLayout(ctrl)

        self.canvas = FigureCanvas(Figure())
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.figure.set_tight_layout(True)
        # White graph background — easier to read traces; panel stays dark
        self.canvas.figure.patch.set_facecolor('#ffffff')
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_facecolor('#ffffff')
        self.ax.tick_params(colors='#444444', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#bbbbbb')
        self.ax.set_xlabel("Time (s)", fontweight='bold', color='#444444')
        self.ax.set_ylabel("Amplitude (a.u.)", fontweight='bold', color='#444444')

        # Interaction state — graph ↔ table two-way sync
        self._trace_lines       = []   # Line2D objects, index-matched to selected_coords
        self._selected_trace    = None # int index of highlighted trace, or None
        self._picked_this_click = False
        self._syncing           = False
        self.canvas.mpl_connect('pick_event',         self._on_graph_pick)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

        self._graph_window = GraphWindow(self)
        self._graph_window.hide()

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
        self.table.itemSelectionChanged.connect(self._on_table_select)

        self._content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._content_splitter.addWidget(self.canvas)
        self._content_splitter.addWidget(self.table)
        self._content_splitter.setStretchFactor(0, 1)
        self._content_splitter.setStretchFactor(1, 1)
        layout.addWidget(self._content_splitter)

        # Force equal split once the dock widget has been shown and sized
        QTimer.singleShot(400, self._init_equal_split)

        self.viewer.mouse_drag_callbacks.append(self.on_click)
        self.viewer.dims.events.current_step.connect(self.update_points_z)

    def apply_theme(self, theme: str = 'dark'):
        self._theme = theme
        self.setStyleSheet(_PANEL_STYLE_LIGHT if theme == 'light' else _PANEL_STYLE_DARK)
        # matplotlib canvas always on white background for readability
        self.canvas.figure.patch.set_facecolor('#ffffff')
        self.ax.set_facecolor('#ffffff')
        self.ax.tick_params(colors='#444444')
        for sp in self.ax.spines.values():
            sp.set_edgecolor('#bbbbbb')
        self.ax.grid(True, color='#e8e8e8', linestyle='--', linewidth=0.5, alpha=0.9)
        if theme == 'light':
            self.btn_random.setStyleSheet(
                "QPushButton { background:#e8ecf8; border:1px solid #c0cce8; color:#2a3560;"
                "  border-radius:4px; padding:4px 10px; font-size:11px; }"
                "QPushButton:hover { background:#d8e0f5; border-color:#7a8cc0; color:#1a2040; }"
            )
            self.btn_clear.setStyleSheet(
                "QPushButton { background:#fde8e8; border:1px solid #e0a0a0; color:#b03030;"
                "  border-radius:4px; padding:4px 10px; font-size:11px; }"
                "QPushButton:hover { background:#fcd0d0; border-color:#c06060; color:#8c2020; }"
            )
        else:
            self.btn_random.setStyleSheet(
                "QPushButton { background:#252840; border:1px solid #3a3e58; color:#c8d0ec;"
                "  border-radius:4px; padding:4px 10px; font-size:11px; }"
                "QPushButton:hover { background:#2d3255; border-color:#5c7cfa; color:#e8ecff; }"
            )
            self.btn_clear.setStyleSheet(
                "QPushButton { background:#2d1a1a; border:1px solid #5c2020; color:#e57373;"
                "  border-radius:4px; padding:4px 10px; font-size:11px; }"
                "QPushButton:hover { background:#3d2020; border-color:#8c3030; color:#ff8a80; }"
            )
        if hasattr(self, '_graph_window') and self._graph_window is not None:
            self._graph_window.apply_theme(theme)
        self.canvas.draw_idle()

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
        amps  = np.nan_to_num(np.max(sigs, axis=1) - np.min(sigs, axis=1), nan=0.0)
        w     = amps ** 2
        total = np.sum(w)
        probs = w / total if total > 0 else None
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

    def _show_graph_popup(self):
        if self._graph_window is None:
            return
        if self.canvas.parent() is not self._graph_window:
            self._graph_window.layout().addWidget(self.canvas)
        n = max(1, len(self.selected_coords))
        self._graph_window.auto_resize(n)
        fname = getattr(self.controls, 'last_path', None)
        if fname:
            self._graph_window.setWindowTitle(os.path.basename(fname))
        self._graph_window.show()
        self._graph_window.raise_()
        self._graph_window.activateWindow()

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
        self._trace_lines    = []
        self._selected_trace = None

        # Re-apply white graph background after clear()
        self.canvas.figure.patch.set_facecolor('#ffffff')
        self.ax.set_facecolor('#ffffff')
        self.ax.tick_params(colors='#444444', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_edgecolor('#bbbbbb')
        self.ax.set_xlabel("Time (s)", fontweight='bold', color='#444444')
        self.ax.set_ylabel("Amplitude (a.u.)", fontweight='bold', color='#444444')
        self.ax.grid(True, color='#e8e8e8', linestyle='--', linewidth=0.5, alpha=0.7)

        # UI #7: show current filename as the plot title
        fname = getattr(self.controls, 'last_path', None)
        if fname:
            self.ax.set_title(os.path.basename(fname), fontsize=9, color='#333333', pad=3)

        if not self.results or not self.selected_coords:
            # UI #6: placeholder text when no cells are selected
            self.ax.text(
                0.5, 0.5,
                "Click on the image to select cells",
                transform=self.ax.transAxes,
                ha='center', va='center',
                fontsize=11, style='italic',
                color='#aaaaaa',
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

            (line,) = self.ax.plot(time, sig, color=color, label=f"P{i+1}",
                                    linewidth=1.2, picker=5)
            self._trace_lines.append(line)

            sig_range_cell = float(np.max(sig) - np.min(sig))
            if sig_range_cell > 1e-6:
                dt_approx = (time[-1] - time[0]) / max(len(time) - 1, 1)
                fps_est   = 1.0 / max(dt_approx, 1e-6)
                cell_peaks, _ = _scipy_find_peaks(
                    sig,
                    prominence=sig_range_cell * 0.10,
                    distance=max(int(fps_est * 0.5), 2),
                )
                if len(cell_peaks) > 0:
                    self.ax.plot(time[cell_peaks], sig[cell_peaks], 'v',
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
                if it is None:
                    it = QTableWidgetItem("")
                    self.table.setItem(r, c_idx, it)
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
            leg = self.ax.legend(loc='upper right', fontsize='small')
            leg.get_frame().set_facecolor('#ffffff')
            leg.get_frame().set_edgecolor('#bbbbbb')
            for text in leg.get_texts():
                text.set_color('#333333')
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

        # Keep graph window title and size in sync whenever traces change
        if hasattr(self, '_graph_window') and self._graph_window is not None:
            fname = getattr(self.controls, 'last_path', None)
            if fname:
                self._graph_window.setWindowTitle(os.path.basename(fname))
            if self._graph_window.isVisible():
                self._graph_window.auto_resize(len(self.selected_coords))

    # ── Graph ↔ Table interaction ────────────────────────────────────────────

    def _on_graph_pick(self, event):
        """Fired when a pickable artist (trace line) is clicked in the graph."""
        if not isinstance(event.artist, Line2D):
            return
        try:
            idx = self._trace_lines.index(event.artist)
        except ValueError:
            return
        self._picked_this_click = True
        self._toggle_highlight(idx)

    def _on_canvas_click(self, event):
        """Fired on every canvas click; deselects if nothing was picked."""
        if not self._picked_this_click:
            self._toggle_highlight(None)
        self._picked_this_click = False

    def _on_table_select(self):
        """Fired when the table row selection changes."""
        if self._syncing:
            return
        rows = self.table.selectionModel().selectedRows()
        idx  = rows[0].row() if rows else None
        self._selected_trace = idx
        self._apply_highlight()

    def _toggle_highlight(self, idx):
        """Select idx, or deselect if it's already selected."""
        if self._selected_trace == idx:
            idx = None
        self._selected_trace = idx
        self._apply_highlight()

    def _apply_highlight(self):
        """Dim all traces except the selected one; sync table row selection."""
        idx = self._selected_trace
        for i, line in enumerate(self._trace_lines):
            if idx is None:
                line.set_alpha(1.0)
                line.set_linewidth(1.2)
            elif i == idx:
                line.set_alpha(1.0)
                line.set_linewidth(2.5)
                line.set_zorder(5)
            else:
                line.set_alpha(0.2)
                line.set_linewidth(1.0)
                line.set_zorder(2)
        self._syncing = True
        if idx is None:
            self.table.clearSelection()
        else:
            self.table.selectRow(idx)
        self._syncing = False
        self.canvas.draw_idle()

    def _init_equal_split(self):
        w = self._content_splitter.width()
        if w > 20:
            half = w // 2
            self._content_splitter.setSizes([half, half])
        else:
            QTimer.singleShot(200, self._init_equal_split)

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
