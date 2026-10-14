"""Sistema de diseño, paleta de colores y estilos QSS para la interfaz de escritorio."""

from __future__ import annotations


DARK_THEME_QSS = """
/* Reset y estilos generales */
QWidget {
    background-color: #0f172a;
    color: #f8fafc;
    font-family: 'Segoe UI', 'Ubuntu', 'Inter', -apple-system, sans-serif;
    font-size: 13px;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
}

/* Paneles y tarjetas */
QFrame#cardPanel, QFrame#sidebarPanel {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px;
}

QGroupBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 14px;
    font-weight: bold;
    color: #38bdf8;
    padding: 14px 10px 10px 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #1e293b;
}

/* Botones principales y secundarios */
QPushButton {
    background-color: #334155;
    color: #f8fafc;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #475569;
    border-color: #64748b;
}

QPushButton:pressed {
    background-color: #1e293b;
}

QPushButton:disabled {
    background-color: #1e293b;
    color: #64748b;
    border-color: #334155;
}

QPushButton#btnPrimary {
    background-color: #0284c7;
    border: 1px solid #0369a1;
    color: #ffffff;
    font-size: 13px;
}

QPushButton#btnPrimary:hover {
    background-color: #0369a1;
}

QPushButton#btnSuccess {
    background-color: #059669;
    border: 1px solid #047857;
    color: #ffffff;
}

QPushButton#btnSuccess:hover {
    background-color: #047857;
}

QPushButton#btnDanger {
    background-color: #dc2626;
    border: 1px solid #b91c1c;
    color: #ffffff;
}

QPushButton#btnDanger:hover {
    background-color: #b91c1c;
}

QPushButton#btnPreset {
    background-color: #1e293b;
    border: 1px solid #475569;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 11px;
}

QPushButton#btnPreset:hover {
    background-color: #334155;
    border-color: #38bdf8;
}

/* Campos de entrada */
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 8px;
    color: #f8fafc;
}

QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #38bdf8;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #334155;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #334155;
    color: #f8fafc;
    selection-background-color: #0284c7;
}

/* Checkboxes y Switches */
QCheckBox {
    spacing: 8px;
    color: #e2e8f0;
    font-weight: 500;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #475569;
    border-radius: 4px;
    background-color: #0f172a;
}

QCheckBox::indicator:checked {
    background-color: #0284c7;
    border-color: #38bdf8;
}

/* Barra de progreso */
QProgressBar {
    border: 1px solid #334155;
    border-radius: 6px;
    text-align: center;
    background-color: #0f172a;
    color: #f8fafc;
    font-weight: 600;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #0284c7;
    border-radius: 5px;
}

/* Tablas */
QTableWidget {
    background-color: #0f172a;
    border: 1px solid #334155;
    gridline-color: #1e293b;
    border-radius: 6px;
}

QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #334155;
    font-weight: 600;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background-color: #0f172a;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #334155;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background-color: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Barra de estado */
QStatusBar {
    background-color: #020617;
    border-top: 1px solid #1e293b;
    color: #94a3b8;
}
"""

LIGHT_THEME_QSS = """
QWidget {
    background-color: #f8fafc;
    color: #0f172a;
    font-family: 'Segoe UI', 'Ubuntu', 'Inter', -apple-system, sans-serif;
    font-size: 13px;
    selection-background-color: #0284c7;
    selection-color: #ffffff;
}

QFrame#cardPanel, QFrame#sidebarPanel {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px;
}

QGroupBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    margin-top: 14px;
    font-weight: bold;
    color: #0284c7;
    padding: 14px 10px 10px 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: #ffffff;
}

QPushButton {
    background-color: #f1f5f9;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #e2e8f0;
}

QPushButton#btnPrimary {
    background-color: #0284c7;
    border: 1px solid #0369a1;
    color: #ffffff;
}

QPushButton#btnSuccess {
    background-color: #059669;
    border: 1px solid #047857;
    color: #ffffff;
}

QPushButton#btnDanger {
    background-color: #dc2626;
    border: 1px solid #b91c1c;
    color: #ffffff;
}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 8px;
    color: #0f172a;
}

QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #0284c7;
}

QProgressBar {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    text-align: center;
    background-color: #e2e8f0;
    color: #0f172a;
    font-weight: 600;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #0284c7;
    border-radius: 5px;
}

QStatusBar {
    background-color: #f1f5f9;
    border-top: 1px solid #e2e8f0;
    color: #64748b;
}
"""


def get_theme_stylesheet(dark: bool = True) -> str:
    """Devuelve la hoja de estilos QSS para el tema deseado."""
    return DARK_THEME_QSS if dark else LIGHT_THEME_QSS
