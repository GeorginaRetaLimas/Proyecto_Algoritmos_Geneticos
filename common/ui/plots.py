"""Widgets de visualización en tiempo real basados en PyQtGraph para los Algoritmos Genéticos."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from common.functions import TargetFunction
from common.stats import GenerationStats

# Configurar PyQtGraph para alto rendimiento y estética moderna
pg.setConfigOption("background", "#0f172a")
pg.setConfigOption("foreground", "#94a3b8")
pg.setConfigOption("antialias", True)


class FunctionPopulationPlotWidget(pg.PlotWidget):
    """Gráfica A: Muestra la curva de la función y la distribución de la población en vivo."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=True)

        self.current_func: TargetFunction | None = None
        self.current_n_vars: int = 1
        self.current_bounds: np.ndarray | None = None

        # Elementos para 1 variable
        # Curva de la función de fondo
        self.curve_func = self.plot(pen=pg.mkPen(color="#475569", width=2.0))
        # Puntos de la población actual
        self.scatter_pop = pg.ScatterPlotItem(
            size=7,
            pen=pg.mkPen(None),
            brush=pg.mkBrush("#38bdf8aa"),
        )
        self.addItem(self.scatter_pop)
        # Mejor individuo de la generación (estrella dorada)
        self.scatter_best = pg.ScatterPlotItem(
            size=14,
            symbol="star",
            pen=pg.mkPen(color="#f59e0b", width=1.5),
            brush=pg.mkBrush("#fbbf24"),
        )
        self.addItem(self.scatter_best)
        # Línea/marcador del óptimo teórico
        self.optimum_line_x = pg.InfiniteLine(
            pen=pg.mkPen(color="#ef4444", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            label="Óptimo {value:.4f}",
            labelOpts={"color": "#ef4444", "position": 0.9, "fill": "#0f172acc"},
        )
        self.addItem(self.optimum_line_x)
        self.optimum_line_x.setVisible(False)

        # Elementos para multivariable (n > 2): gráfico de barras
        self.bars_best = pg.BarGraphItem(x=[], height=[], width=0.35, brush="#38bdf8")
        self.bars_opt = pg.BarGraphItem(x=[], height=[], width=0.35, brush="#ef4444aa")
        self.addItem(self.bars_best)
        self.addItem(self.bars_opt)
        self.bars_best.setVisible(False)
        self.bars_opt.setVisible(False)

        self.setTitle("<span style='color: #38bdf8; font-weight: bold;'>Gráfica A — Función y Población</span>", size="11pt")
        self.setLabel("bottom", "Fenotipo (x)")
        self.setLabel("left", "Aptitud bruta f(x)")

    def set_function_context(self, func: TargetFunction, n_vars: int, bounds: np.ndarray) -> None:
        """Inicializa el contexto de la función y dibuja la curva de fondo si n=1."""
        self.current_func = func
        self.current_n_vars = n_vars
        self.current_bounds = bounds

        # Limpiar visibilidad de elementos según dimensión
        if n_vars == 1:
            self.curve_func.setVisible(True)
            self.scatter_pop.setVisible(True)
            self.scatter_best.setVisible(True)
            self.bars_best.setVisible(False)
            self.bars_opt.setVisible(False)

            self.setLabel("bottom", "Variable x")
            self.setLabel("left", "Aptitud f(x)")

            # Calcular curva analítica con 800 puntos
            x_min, x_max = bounds[0]
            xs = np.linspace(x_min, x_max, 800)
            ys = func.evaluate(xs.reshape(-1, 1))
            self.curve_func.setData(xs, ys)

            # Óptimo teórico
            opt_x = func.get_optimum_x(1)[0]
            self.optimum_line_x.setValue(opt_x)
            self.optimum_line_x.setVisible(True)

            self.enableAutoRange()
        elif n_vars == 2:
            self.curve_func.setVisible(False)
            self.curve_func.clear()
            self.scatter_pop.setVisible(True)
            self.scatter_best.setVisible(True)
            self.bars_best.setVisible(False)
            self.bars_opt.setVisible(False)
            self.optimum_line_x.setVisible(False)

            self.setLabel("bottom", "Variable x₁")
            self.setLabel("left", "Variable x₂")
            self.enableAutoRange()
        else:
            # n > 2: Visualizar vector de coordenadas del mejor vs óptimo teórico
            self.curve_func.setVisible(False)
            self.curve_func.clear()
            self.scatter_pop.setVisible(False)
            self.scatter_best.setVisible(False)
            self.optimum_line_x.setVisible(False)
            self.bars_best.setVisible(True)
            self.bars_opt.setVisible(True)

            self.setLabel("bottom", "Índice de variable (1..n)")
            self.setLabel("left", "Valor de coordenada")

    def update_generation(self, stats: GenerationStats) -> None:
        """Actualiza los individuos de la población y el mejor individuo en la gráfica."""
        if self.current_n_vars == 1:
            pop_x = stats.population_x[:, 0]
            pop_y = stats.population_fitness_raw
            self.scatter_pop.setData(x=pop_x, y=pop_y)

            best_x = float(stats.best_x[0])
            best_y = float(stats.best_fitness_raw)
            self.scatter_best.setData(x=[best_x], y=[best_y])
        elif self.current_n_vars == 2:
            x1 = stats.population_x[:, 0]
            x2 = stats.population_x[:, 1]
            self.scatter_pop.setData(x=x1, y=x2)

            best_x1 = float(stats.best_x[0])
            best_x2 = float(stats.best_x[1])
            self.scatter_best.setData(x=[best_x1], y=[best_x2])
        else:
            # n > 2
            n = self.current_n_vars
            indices = np.arange(1, n + 1)
            best_vec = stats.best_x
            if self.current_func is not None:
                opt_vec = self.current_func.get_optimum_x(n)
            else:
                opt_vec = np.zeros(n)

            self.bars_best.setOpts(x=indices - 0.2, height=best_vec, width=0.35)
            self.bars_opt.setOpts(x=indices + 0.2, height=opt_vec, width=0.35)

    def export_to_png(self, filepath: str | Path) -> None:
        """Exporta la gráfica actual a un archivo PNG."""
        pixmap = self.grab()
        pixmap.save(str(filepath), "PNG")


class ConvergencePlotWidget(pg.PlotWidget):
    """Gráfica B: Muestra las curvas de convergencia (mejor, media, peor, óptimo) en vivo."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=True, y=True)
        self.addLegend(offset=(10, 10), brush="#1e293b99", pen="#334155")

        self.setTitle("<span style='color: #38bdf8; font-weight: bold;'>Gráfica B — Convergencia</span>", size="11pt")
        self.setLabel("bottom", "Generación")
        self.setLabel("left", "Aptitud bruta f(x)")

        # Curvas de datos
        self.curve_best_so_far = self.plot(
            pen=pg.mkPen(color="#38bdf8", width=2.5),
            name="Mejor histórico",
        )
        self.curve_gen_best = self.plot(
            pen=pg.mkPen(color="#10b981", width=1.8),
            name="Mejor de generación",
        )
        self.curve_mean = self.plot(
            pen=pg.mkPen(color="#f59e0b", width=1.5, style=QtCore.Qt.PenStyle.DashLine),
            name="Media de población",
        )
        self.curve_worst = self.plot(
            pen=pg.mkPen(color="#64748b", width=1.0, style=QtCore.Qt.PenStyle.DotLine),
            name="Peor de generación",
        )
        self.line_optimum = pg.InfiniteLine(
            angle=0,
            pen=pg.mkPen(color="#ef4444", width=1.8, style=QtCore.Qt.PenStyle.DashLine),
            label="Óptimo {value:.4f}",
            labelOpts={"color": "#ef4444", "position": 0.85, "fill": "#0f172acc"},
        )
        self.addItem(self.line_optimum)

        # Buffers de historial
        self.history_gens: list[int] = []
        self.history_best_so_far: list[float] = []
        self.history_gen_best: list[float] = []
        self.history_mean: list[float] = []
        self.history_worst: list[float] = []

    def set_optimum(self, optimum_value: float) -> None:
        """Establece la línea de referencia horizontal del óptimo teórico."""
        self.line_optimum.setValue(optimum_value)
        self.line_optimum.setVisible(True)

    def reset(self) -> None:
        """Limpia el historial de convergencia."""
        self.history_gens.clear()
        self.history_best_so_far.clear()
        self.history_gen_best.clear()
        self.history_mean.clear()
        self.history_worst.clear()

        self.curve_best_so_far.clear()
        self.curve_gen_best.clear()
        self.curve_mean.clear()
        self.curve_worst.clear()

    def update_generation(self, stats: GenerationStats) -> None:
        """Agrega los datos de la generación actual a las curvas."""
        self.history_gens.append(stats.generation)
        self.history_best_so_far.append(stats.best_so_far_raw)
        self.history_gen_best.append(stats.best_fitness_raw)
        self.history_mean.append(stats.mean_fitness_raw)
        self.history_worst.append(stats.worst_fitness_raw)

        gens = np.array(self.history_gens)
        self.curve_best_so_far.setData(gens, np.array(self.history_best_so_far))
        self.curve_gen_best.setData(gens, np.array(self.history_gen_best))
        self.curve_mean.setData(gens, np.array(self.history_mean))
        self.curve_worst.setData(gens, np.array(self.history_worst))

    def export_to_png(self, filepath: str | Path) -> None:
        """Exporta la gráfica de convergencia a un archivo PNG."""
        pixmap = self.grab()
        pixmap.save(str(filepath), "PNG")
