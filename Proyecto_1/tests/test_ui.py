"""Pruebas básicas de inicialización de la interfaz gráfica (Proyecto 1)."""

import os
import pytest
from PySide6 import QtWidgets

# Configurar plataforma offscreen para entornos sin servidor gráfico directo en CI/tests
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_main_window_initialization(qapp):
    from Proyecto_1.app import MainWindow

    window = MainWindow()
    assert window.windowTitle().startswith("Algoritmos Genéticos")
    assert window.plot_function is not None
    assert window.plot_convergence is not None
    assert window.current_func.id == "M2"

    # Verificar cambio de función
    window._on_function_changed("M1")
    assert window.current_func.id == "M1"
    assert window.spin_xmin.value() == 0.0
    assert window.spin_xmax.value() == 1.0

    # Verificar cambio a función multivariable
    window._on_function_changed("V2")
    assert window.current_func.id == "V2"
    assert window.spin_n_vars.isEnabled()

    # Verificar cambio de tema
    initial_theme = window.is_dark_theme
    window._toggle_theme()
    assert window.is_dark_theme != initial_theme
    window._toggle_theme()
    assert window.is_dark_theme == initial_theme
