"""Pruebas básicas de la interfaz gráfica del Proyecto 2 (offscreen)."""

import os
import pytest
from PySide6 import QtWidgets

os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_main_window_real_initialization(qapp):
    from Proyecto_2.app import MainWindowReal

    window = MainWindowReal()
    window.show()
    assert window.windowTitle().startswith("Algoritmos Genéticos — Proyecto 2")
    assert window.plot_function is not None
    assert window.plot_convergence is not None

    # Probar selector de operadores
    for i in range(window.combo_crossover.count()):
        window.combo_crossover.setCurrentIndex(i)
        assert window.combo_crossover.currentData() in ["sbx", "arithmetic", "discrete_uniform", "one_point"]

    # Probar advertencia de 1 variable con 'one_point'
    window._on_function_changed("M1")  # n=1
    idx_op = window.combo_crossover.findData("one_point")
    window.combo_crossover.setCurrentIndex(idx_op)
    assert window.lbl_warning_one_point.isVisible()

    # Cambiar a multivariable V2
    window._on_function_changed("V2")  # n=5
    assert not window.lbl_warning_one_point.isVisible()

    # Probar toggle de tema
    t0 = window.is_dark_theme
    window._toggle_theme()
    assert window.is_dark_theme != t0
