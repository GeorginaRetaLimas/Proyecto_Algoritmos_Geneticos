"""Aplicación principal con interfaz gráfica moderna para el Proyecto 1 (AG binario).

Permite configurar interactivamente los parámetros, visualizar la población y la curva
de la función en cada generación, seguir la convergencia en tiempo real, controlar la
velocidad de la animación (con preset de 10 min), pausar, avanzar paso a paso y exportar.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from common.functions import ALL_FUNCTIONS, CATALOG, TargetFunction, get_function
from common.stats import GenerationStats, export_history_to_csv
from common.ui.plots import ConvergencePlotWidget, FunctionPopulationPlotWidget
from common.ui.theme import get_theme_stylesheet
from common.ui.worker import GeneticAlgorithmWorker
from Proyecto_1.ga_binario import BinaryGAConfig, BinaryGeneticAlgorithm


class MainWindow(QtWidgets.QMainWindow):
    """Ventana principal del Proyecto 1: Algoritmo Genético binario."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Algoritmos Genéticos — Proyecto 1: Optimización de Funciones (Representación Binaria)")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 750)

        # Estado interno
        self.is_dark_theme = True
        self.worker: GeneticAlgorithmWorker | None = None
        self.current_func: TargetFunction = get_function("M2")
        self.history: list[GenerationStats] = []
        self.start_time: float = 0.0

        # Configurar hoja de estilos inicial
        self.setStyleSheet(get_theme_stylesheet(dark=True))

        # Construir UI
        self._init_ui()

        # Cargar valores por defecto de la función inicial
        self._on_function_changed("M2")

    def _init_ui(self) -> None:
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # 1. PANEL IZQUIERDO: Configuración y controles (Sidebar)
        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(380)
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        sidebar_content = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(0, 0, 8, 0)
        sidebar_layout.setSpacing(12)

        # Group 1: Selección de función
        sidebar_layout.addWidget(self._create_function_group())

        # Group 2: Parámetros del AG
        sidebar_layout.addWidget(self._create_ga_params_group())

        # Group 3: Control de animación y velocidad
        sidebar_layout.addWidget(self._create_animation_control_group())

        # Group 4: Botones de acción y exportación
        sidebar_layout.addWidget(self._create_actions_group())

        sidebar_layout.addStretch()
        sidebar_scroll.setWidget(sidebar_content)
        main_layout.addWidget(sidebar_scroll)

        # 2. PANEL CENTRAL: Gráficas en vivo (A y B)
        graphs_layout = QtWidgets.QVBoxLayout()
        graphs_layout.setSpacing(12)

        self.plot_function = FunctionPopulationPlotWidget()
        self.plot_convergence = ConvergencePlotWidget()

        graphs_layout.addWidget(self.plot_function, stretch=1)
        graphs_layout.addWidget(self.plot_convergence, stretch=1)

        main_layout.addLayout(graphs_layout, stretch=3)

        # 3. PANEL DERECHO: Resultados en vivo y telemetría
        right_panel = self._create_results_panel()
        main_layout.addWidget(right_panel, stretch=1)

        # Barra de estado
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo. Seleccione una función y pulse 'Iniciar Evolución'.")

    # -----------------------------------------------------------------------
    # Componentes de la UI
    # -----------------------------------------------------------------------

    def _create_function_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("1. Función Objetivo")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        # Filtro de categoría
        cat_layout = QtWidgets.QHBoxLayout()
        cat_label = QtWidgets.QLabel("Categoría:")
        self.combo_category = QtWidgets.QComboBox()
        self.combo_category.addItems(["Todas", "Maximización 1D", "Minimización 1D", "Varias variables"])
        self.combo_category.currentIndexChanged.connect(self._on_category_filter_changed)
        cat_layout.addWidget(cat_label)
        cat_layout.addWidget(self.combo_category)
        layout.addLayout(cat_layout)

        # Selector de función
        func_layout = QtWidgets.QHBoxLayout()
        func_label = QtWidgets.QLabel("Función:")
        self.combo_function = QtWidgets.QComboBox()
        for f in ALL_FUNCTIONS:
            self.combo_function.addItem(f"{f.id} — {f.name}", userData=f.id)
        self.combo_function.currentIndexChanged.connect(self._on_combo_function_selected)
        func_layout.addWidget(func_label)
        func_layout.addWidget(self.combo_function)
        layout.addLayout(func_layout)

        # Tarjeta informativa de la función
        self.card_formula = QtWidgets.QFrame()
        self.card_formula.setObjectName("cardPanel")
        card_layout = QtWidgets.QVBoxLayout(self.card_formula)
        card_layout.setContentsMargins(8, 8, 8, 8)

        self.lbl_formula_display = QtWidgets.QLabel("f(x) = ...")
        self.lbl_formula_display.setWordWrap(True)
        self.lbl_formula_display.setStyleSheet("color: #38bdf8; font-weight: bold;")
        card_layout.addWidget(self.lbl_formula_display)

        self.lbl_optimum_info = QtWidgets.QLabel("Óptimo teórico: ...")
        self.lbl_optimum_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        card_layout.addWidget(self.lbl_optimum_info)

        layout.addWidget(self.card_formula)
        return group

    def _create_ga_params_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("2. Parámetros del AG")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        form_layout = QtWidgets.QGridLayout()
        form_layout.setSpacing(8)

        # Dominio [x_min, x_max]
        form_layout.addWidget(QtWidgets.QLabel("Intervalo [x_min, x_max]:"), 0, 0)
        bounds_layout = QtWidgets.QHBoxLayout()
        self.spin_xmin = QtWidgets.QDoubleSpinBox()
        self.spin_xmin.setRange(-100000.0, 100000.0)
        self.spin_xmin.setDecimals(4)
        self.spin_xmin.valueChanged.connect(self._update_chromosome_info)

        self.spin_xmax = QtWidgets.QDoubleSpinBox()
        self.spin_xmax.setRange(-100000.0, 100000.0)
        self.spin_xmax.setDecimals(4)
        self.spin_xmax.valueChanged.connect(self._update_chromosome_info)

        bounds_layout.addWidget(self.spin_xmin)
        bounds_layout.addWidget(QtWidgets.QLabel("a"))
        bounds_layout.addWidget(self.spin_xmax)
        form_layout.addLayout(bounds_layout, 0, 1)

        # Número de variables (n)
        form_layout.addWidget(QtWidgets.QLabel("Variables (n):"), 1, 0)
        self.spin_n_vars = QtWidgets.QSpinBox()
        self.spin_n_vars.setRange(1, 30)
        self.spin_n_vars.setValue(1)
        self.spin_n_vars.valueChanged.connect(self._on_n_vars_changed)
        form_layout.addWidget(self.spin_n_vars, 1, 1)

        # Tolerancia (TOL)
        form_layout.addWidget(QtWidgets.QLabel("Tolerancia (TOL):"), 2, 0)
        self.edit_tol = QtWidgets.QLineEdit("0.0001")
        self.edit_tol.textChanged.connect(self._update_chromosome_info)
        form_layout.addWidget(self.edit_tol, 2, 1)

        # Tamaño de población
        form_layout.addWidget(QtWidgets.QLabel("Población (tam_pob):"), 3, 0)
        self.spin_tam_pob = QtWidgets.QSpinBox()
        self.spin_tam_pob.setRange(10, 5000)
        self.spin_tam_pob.setValue(500)
        self.spin_tam_pob.setSingleStep(50)
        form_layout.addWidget(self.spin_tam_pob, 3, 1)

        # Número de generaciones
        form_layout.addWidget(QtWidgets.QLabel("Generaciones:"), 4, 0)
        self.spin_num_gen = QtWidgets.QSpinBox()
        self.spin_num_gen.setRange(10, 20000)
        self.spin_num_gen.setValue(1000)
        self.spin_num_gen.setSingleStep(50)
        self.spin_num_gen.valueChanged.connect(self._update_delay_from_duration)
        form_layout.addWidget(self.spin_num_gen, 4, 1)

        # Probabilidad de cruce
        form_layout.addWidget(QtWidgets.QLabel("Prob. Cruce:"), 5, 0)
        self.spin_prob_cruce = QtWidgets.QDoubleSpinBox()
        self.spin_prob_cruce.setRange(0.0, 1.0)
        self.spin_prob_cruce.setSingleStep(0.05)
        self.spin_prob_cruce.setDecimals(2)
        self.spin_prob_cruce.setValue(0.40)
        form_layout.addWidget(self.spin_prob_cruce, 5, 1)

        # Probabilidad de mutación por bit
        form_layout.addWidget(QtWidgets.QLabel("Prob. Mutación (bit):"), 6, 0)
        self.spin_prob_mut = QtWidgets.QDoubleSpinBox()
        self.spin_prob_mut.setRange(0.0, 1.0)
        self.spin_prob_mut.setSingleStep(0.0005)
        self.spin_prob_mut.setDecimals(4)
        self.spin_prob_mut.setValue(0.0010)
        form_layout.addWidget(self.spin_prob_mut, 6, 1)

        # Semilla RNG
        form_layout.addWidget(QtWidgets.QLabel("Semilla RNG:"), 7, 0)
        seed_layout = QtWidgets.QHBoxLayout()
        self.edit_seed = QtWidgets.QLineEdit("")
        self.edit_seed.setPlaceholderText("Aleatoria")
        self.btn_new_seed = QtWidgets.QPushButton("🎲")
        self.btn_new_seed.setFixedWidth(36)
        self.btn_new_seed.clicked.connect(self._generate_random_seed)
        seed_layout.addWidget(self.edit_seed)
        seed_layout.addWidget(self.btn_new_seed)
        form_layout.addLayout(seed_layout, 7, 1)

        layout.addLayout(form_layout)

        # Switch de elitismo y porcentaje
        elite_layout = QtWidgets.QHBoxLayout()
        self.chk_elitism = QtWidgets.QCheckBox("Elitismo activado")
        self.chk_elitism.setChecked(True)
        self.spin_pct_elite = QtWidgets.QDoubleSpinBox()
        self.spin_pct_elite.setRange(0.1, 50.0)
        self.spin_pct_elite.setSingleStep(0.5)
        self.spin_pct_elite.setValue(2.0)
        self.spin_pct_elite.setSuffix(" %")
        elite_layout.addWidget(self.chk_elitism)
        elite_layout.addWidget(self.spin_pct_elite)
        layout.addLayout(elite_layout)

        # Tarjeta informativa de longitud de cromosoma
        self.lbl_chromosome_info = QtWidgets.QLabel("Cromosoma: ...")
        self.lbl_chromosome_info.setStyleSheet("color: #a7f3d0; font-size: 11px; padding: 4px; background: #064e3b55; border-radius: 4px;")
        layout.addWidget(self.lbl_chromosome_info)

        return group

    def _create_animation_control_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("3. Velocidad de Animación")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        # Presets requeridos por la especificación (§6)
        preset_layout = QtWidgets.QHBoxLayout()
        self.btn_preset_10min = QtWidgets.QPushButton("10 Min (Demo)")
        self.btn_preset_10min.setObjectName("btnPreset")
        self.btn_preset_10min.clicked.connect(lambda: self._set_target_duration(600.0))

        self.btn_preset_1min = QtWidgets.QPushButton("1 Min")
        self.btn_preset_1min.setObjectName("btnPreset")
        self.btn_preset_1min.clicked.connect(lambda: self._set_target_duration(60.0))

        self.btn_preset_10s = QtWidgets.QPushButton("10 s")
        self.btn_preset_10s.setObjectName("btnPreset")
        self.btn_preset_10s.clicked.connect(lambda: self._set_target_duration(10.0))

        self.btn_preset_max = QtWidgets.QPushButton("Máx. Vel.")
        self.btn_preset_max.setObjectName("btnPreset")
        self.btn_preset_max.clicked.connect(lambda: self._set_target_duration(0.0))

        preset_layout.addWidget(self.btn_preset_10min)
        preset_layout.addWidget(self.btn_preset_1min)
        preset_layout.addWidget(self.btn_preset_10s)
        preset_layout.addWidget(self.btn_preset_max)
        layout.addLayout(preset_layout)

        # Slider de retardo en vivo
        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("Retardo:"))
        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_speed.setRange(0, 1000)  # en milisegundos
        self.slider_speed.setValue(30)
        self.slider_speed.valueChanged.connect(self._on_speed_slider_changed)
        slider_layout.addWidget(self.slider_speed)

        self.lbl_speed_val = QtWidgets.QLabel("30 ms/gen")
        self.lbl_speed_val.setFixedWidth(75)
        slider_layout.addWidget(self.lbl_speed_val)
        layout.addLayout(slider_layout)

        self.lbl_duration_calc = QtWidgets.QLabel("Duración estimada: ~30 s")
        self.lbl_duration_calc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_duration_calc)

        return group

    def _create_actions_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("4. Controles de Ejecución")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        # Botón principal Iniciar / Reanudar
        self.btn_start = QtWidgets.QPushButton("▶ Iniciar Evolución")
        self.btn_start.setObjectName("btnPrimary")
        self.btn_start.setFixedHeight(36)
        self.btn_start.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.btn_start)

        # Botones de control secundario (Pausa, Paso a paso, Detener)
        controls_layout = QtWidgets.QHBoxLayout()
        self.btn_pause = QtWidgets.QPushButton("⏸ Pausar")
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self._on_pause_clicked)

        self.btn_step = QtWidgets.QPushButton("⏭ Paso")
        self.btn_step.setEnabled(False)
        self.btn_step.clicked.connect(self._on_step_clicked)

        self.btn_stop = QtWidgets.QPushButton("⏹ Detener")
        self.btn_stop.setObjectName("btnDanger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)

        controls_layout.addWidget(self.btn_pause)
        controls_layout.addWidget(self.btn_step)
        controls_layout.addWidget(self.btn_stop)
        layout.addLayout(controls_layout)

        # Botón reiniciar
        self.btn_reset = QtWidgets.QPushButton("🔄 Reiniciar")
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self.btn_reset)

        # Botones de exportación
        export_layout = QtWidgets.QHBoxLayout()
        self.btn_export_png = QtWidgets.QPushButton("📷 Guardar PNG")
        self.btn_export_png.clicked.connect(self._on_export_png_clicked)

        self.btn_export_csv = QtWidgets.QPushButton("📊 Guardar CSV")
        self.btn_export_csv.clicked.connect(self._on_export_csv_clicked)

        export_layout.addWidget(self.btn_export_png)
        export_layout.addWidget(self.btn_export_csv)
        layout.addLayout(export_layout)

        return group

    def _create_results_panel(self) -> QtWidgets.QFrame:
        panel = QtWidgets.QFrame()
        panel.setObjectName("sidebarPanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Encabezado con interruptor de tema
        header_layout = QtWidgets.QHBoxLayout()
        title_lbl = QtWidgets.QLabel("Telemetría")
        title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        self.btn_toggle_theme = QtWidgets.QPushButton("🌓 Tema")
        self.btn_toggle_theme.setFixedWidth(70)
        self.btn_toggle_theme.clicked.connect(self._toggle_theme)
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_toggle_theme)
        layout.addLayout(header_layout)

        # Barra de progreso de generaciones
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Gen 0 / 1000")
        layout.addWidget(self.progress_bar)

        # Tarjetas KPI
        # 1. Mejor Valor Actual
        self.card_best = self._create_kpi_card("Mejor Aptitud f(x)", "--", color="#10b981")
        layout.addWidget(self.card_best)

        # 2. Óptimo y Error
        self.card_error = self._create_kpi_card("Error Absoluto (|f - f*|)", "--", color="#38bdf8")
        layout.addWidget(self.card_error)

        # 3. Badge Estado de Éxito
        self.lbl_status_badge = QtWidgets.QLabel("LISTO")
        self.lbl_status_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_status_badge.setStyleSheet("background-color: #334155; color: #ffffff; padding: 6px; border-radius: 6px; font-weight: bold;")
        layout.addWidget(self.lbl_status_badge)

        # 4. Mejor Fenotipo x
        self.lbl_best_x_title = QtWidgets.QLabel("Mejor Fenotipo (x):")
        self.lbl_best_x_title.setStyleSheet("font-weight: 600; color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_best_x_title)

        self.txt_best_x = QtWidgets.QTextEdit()
        self.txt_best_x.setReadOnly(True)
        self.txt_best_x.setFixedHeight(65)
        self.txt_best_x.setPlaceholderText("Coordenadas de la mejor solución...")
        layout.addWidget(self.txt_best_x)

        # 5. Tiempo y FPS
        self.lbl_time_info = QtWidgets.QLabel("Tiempo: 0.0 s | Semilla: --")
        self.lbl_time_info.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(self.lbl_time_info)

        # 6. Tabla de historial reciente
        layout.addWidget(QtWidgets.QLabel("Últimas generaciones:"))
        self.table_history = QtWidgets.QTableWidget()
        self.table_history.setColumnCount(4)
        self.table_history.setHorizontalHeaderLabels(["Gen", "Mejor", "Media", "Error"])
        self.table_history.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_history.verticalHeader().setVisible(False)
        self.table_history.setRowCount(0)
        layout.addWidget(self.table_history)

        return panel

    def _create_kpi_card(self, title: str, value: str, color: str = "#38bdf8") -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("cardPanel")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(8, 8, 8, 8)

        lbl_t = QtWidgets.QLabel(title)
        lbl_t.setStyleSheet("font-size: 11px; color: #94a3b8; font-weight: 600;")
        lbl_v = QtWidgets.QLabel(value)
        lbl_v.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")

        layout.addWidget(lbl_t)
        layout.addWidget(lbl_v)
        card.lbl_val = lbl_v  # Guardar referencia dinámica
        return card

    # -----------------------------------------------------------------------
    # Manejadores de eventos y lógica interactiva
    # -----------------------------------------------------------------------

    def _toggle_theme(self) -> None:
        self.is_dark_theme = not self.is_dark_theme
        self.setStyleSheet(get_theme_stylesheet(dark=self.is_dark_theme))

    def _on_category_filter_changed(self, index: int) -> None:
        category_map = {
            1: "max_1d",
            2: "min_1d",
            3: "multi",
        }
        target_cat = category_map.get(index)

        self.combo_function.blockSignals(True)
        self.combo_function.clear()
        for f in ALL_FUNCTIONS:
            if target_cat is None or f.category == target_cat:
                self.combo_function.addItem(f"{f.id} — {f.name}", userData=f.id)
        self.combo_function.blockSignals(False)

        if self.combo_function.count() > 0:
            self._on_combo_function_selected(0)

    def _on_combo_function_selected(self, index: int) -> None:
        if index < 0:
            return
        func_id = self.combo_function.itemData(index)
        if func_id:
            self._on_function_changed(func_id)

    def _on_function_changed(self, func_id: str) -> None:
        self.current_func = get_function(func_id)
        f = self.current_func

        # Actualizar tarjeta de fórmula
        self.lbl_formula_display.setText(f.formula_display)
        target_type_str = "MAXIMIZACIÓN" if f.target_type == "max" else "MINIMIZACIÓN"

        # Configurar límites e intervalo
        x_min, x_max = f.bounds
        self.spin_xmin.blockSignals(True)
        self.spin_xmax.blockSignals(True)
        self.spin_xmin.setValue(x_min)
        self.spin_xmax.setValue(x_max)
        self.spin_xmin.blockSignals(False)
        self.spin_xmax.blockSignals(False)

        # Configurar dimensión de variables
        self.spin_n_vars.blockSignals(True)
        self.spin_n_vars.setValue(f.default_n_vars)
        self.spin_n_vars.setEnabled(f.is_n_configurable)
        self.spin_n_vars.blockSignals(False)

        n_vars = self.spin_n_vars.value()
        opt_val = f.get_optimum_val(n_vars)
        self.lbl_optimum_info.setText(f"Tipo: {target_type_str} | Óptimo teórico: {opt_val:.4f}")

        # Actualizar gráficas
        bounds_arr = f.get_bounds_array(n_vars)
        bounds_arr[:, 0] = self.spin_xmin.value()
        bounds_arr[:, 1] = self.spin_xmax.value()

        self.plot_function.set_function_context(f, n_vars, bounds_arr)
        self.plot_convergence.set_optimum(opt_val)
        self.plot_convergence.reset()

        self._update_chromosome_info()
        self._update_delay_from_duration()

    def _on_n_vars_changed(self, val: int) -> None:
        if self.current_func is not None:
            opt_val = self.current_func.get_optimum_val(val)
            self.lbl_optimum_info.setText(
                f"Tipo: {'MAXIMIZACIÓN' if self.current_func.target_type == 'max' else 'MINIMIZACIÓN'} | Óptimo: {opt_val:.4f}"
            )
            bounds_arr = self.current_func.get_bounds_array(val)
            self.plot_function.set_function_context(self.current_func, val, bounds_arr)
            self.plot_convergence.set_optimum(opt_val)
            self._update_chromosome_info()

    def _update_chromosome_info(self) -> None:
        try:
            tol = float(self.edit_tol.text().strip())
            tol = max(tol, 1e-12)
        except ValueError:
            tol = 1e-4

        x_min = self.spin_xmin.value()
        x_max = self.spin_xmax.value()
        n_vars = self.spin_n_vars.value()

        if x_max <= x_min:
            self.lbl_chromosome_info.setText("Error: x_max debe ser estrictamente mayor que x_min.")
            self.lbl_chromosome_info.setStyleSheet("color: #fca5a5; background: #7f1d1d55; border-radius: 4px; padding: 4px;")
            return

        ratio = 1.0 + (x_max - x_min) / tol
        lcrom = int(np.ceil(np.log2(max(ratio, 1.0))))
        total_bits = lcrom * n_vars

        self.lbl_chromosome_info.setText(f"Longitud: {lcrom} bits/var × {n_vars} var = {total_bits} bits totales")
        self.lbl_chromosome_info.setStyleSheet("color: #a7f3d0; background: #064e3b55; border-radius: 4px; padding: 4px;")

    def _generate_random_seed(self) -> None:
        new_seed = int(np.random.default_rng().integers(1, 999999))
        self.edit_seed.setText(str(new_seed))

    def _set_target_duration(self, seconds: float) -> None:
        num_gen = max(1, self.spin_num_gen.value())
        if seconds <= 0.0:
            delay_ms = 0
        else:
            delay_ms = int(round((seconds / num_gen) * 1000.0))

        self.slider_speed.setValue(delay_ms)

    def _update_delay_from_duration(self) -> None:
        delay_ms = self.slider_speed.value()
        self._on_speed_slider_changed(delay_ms)

    def _on_speed_slider_changed(self, value: int) -> None:
        delay_seconds = value / 1000.0
        num_gen = max(1, self.spin_num_gen.value())
        total_est = delay_seconds * num_gen

        if value == 0:
            self.lbl_speed_val.setText("Máx Vel")
            self.lbl_duration_calc.setText("Duración estimada: casi instantáneo")
        else:
            self.lbl_speed_val.setText(f"{value} ms/gen")
            if total_est < 60:
                self.lbl_duration_calc.setText(f"Duración estimada: ~{total_est:.1f} s")
            else:
                self.lbl_duration_calc.setText(f"Duración estimada: ~{total_est / 60:.1f} min")

        if self.worker is not None:
            self.worker.set_delay(delay_seconds)

    # -----------------------------------------------------------------------
    # Ejecución del Algoritmo Genético
    # -----------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        # Si ya hay un worker corriendo y pausado, reanudar
        if self.worker is not None and self.worker.isRunning():
            self.worker.resume()
            self.btn_start.setText("▶ Reanudado")
            self.btn_pause.setEnabled(True)
            self.btn_step.setEnabled(True)
            self.btn_stop.setEnabled(True)
            return

        # Validaciones
        x_min = self.spin_xmin.value()
        x_max = self.spin_xmax.value()
        if x_max <= x_min:
            QtWidgets.QMessageBox.warning(self, "Parámetros inválidos", "El límite superior x_max debe ser mayor que x_min.")
            return

        try:
            tol = float(self.edit_tol.text().strip())
            if tol <= 0:
                raise ValueError
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Tolerancia inválida", "La tolerancia TOL debe ser un número positivo.")
            return

        seed_text = self.edit_seed.text().strip()
        seed_val = int(seed_text) if seed_text.isdigit() else None

        n_vars = self.spin_n_vars.value()
        num_gen = self.spin_num_gen.value()
        tam_pob = self.spin_tam_pob.value()
        prob_cruce = self.spin_prob_cruce.value()
        prob_mut = self.spin_prob_mut.value()
        elitismo = self.chk_elitism.isChecked()
        pct_elite = self.spin_pct_elite.value()

        bounds_arr = np.tile(np.array([x_min, x_max], dtype=np.float64), (n_vars, 1))

        # Construir configuración
        config = BinaryGAConfig(
            function=self.current_func,
            n_vars=n_vars,
            bounds=bounds_arr,
            tol=tol,
            tam_pob=tam_pob,
            num_gen=num_gen,
            prob_cruce=prob_cruce,
            prob_mut=prob_mut,
            elitismo=elitismo,
            pct_elite=pct_elite,
            seed=seed_val,
        )

        # Crear motor del AG
        ga = BinaryGeneticAlgorithm(config)
        self.edit_seed.setText(str(ga.used_seed))

        # Preparar gráficas y paneles
        self.history.clear()
        self.table_history.setRowCount(0)
        self.progress_bar.setRange(0, num_gen)
        self.progress_bar.setValue(0)
        self.plot_convergence.reset()
        self.plot_function.set_function_context(self.current_func, n_vars, bounds_arr)

        delay_seconds = self.slider_speed.value() / 1000.0

        # Iniciar worker en hilo separado
        self.worker = GeneticAlgorithmWorker(ga.run(), delay_seconds=delay_seconds)
        self.worker.generation_updated.connect(self._on_generation_updated)
        self.worker.evolution_finished.connect(self._on_evolution_finished)
        self.worker.status_changed.connect(self._on_worker_status_changed)
        self.worker.error_occurred.connect(self._on_worker_error)

        self.start_time = time.perf_counter()
        self.worker.start()

        # Ajustar botones
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_step.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.status_bar.showMessage("Evolución en curso...")

    def _on_pause_clicked(self) -> None:
        if self.worker is not None:
            self.worker.pause()
            self.btn_start.setEnabled(True)
            self.btn_start.setText("▶ Reanudar")
            self.btn_pause.setEnabled(False)

    def _on_step_clicked(self) -> None:
        if self.worker is not None:
            self.worker.step()
            self.btn_start.setEnabled(True)
            self.btn_start.setText("▶ Reanudar")

    def _on_stop_clicked(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait()
            self._on_evolution_finished()

    def _on_reset_clicked(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        self.history.clear()
        self.plot_convergence.reset()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Gen 0 / {self.spin_num_gen.value()}")
        self.card_best.lbl_val.setText("--")
        self.card_error.lbl_val.setText("--")
        self.txt_best_x.clear()
        self.table_history.setRowCount(0)
        self.lbl_status_badge.setText("REINICIADO")
        self.lbl_status_badge.setStyleSheet("background-color: #334155; color: #ffffff; padding: 6px; border-radius: 6px; font-weight: bold;")

        self.btn_start.setEnabled(True)
        self.btn_start.setText("▶ Iniciar Evolución")
        self.btn_pause.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Listo para nueva ejecución.")

    def _on_worker_status_changed(self, msg: str) -> None:
        self.status_bar.showMessage(msg)

    def _on_worker_error(self, err: str) -> None:
        QtWidgets.QMessageBox.critical(self, "Error en ejecución", f"Ocurrió un error en el motor del AG: {err}")
        self._on_evolution_finished()

    def _on_generation_updated(self, stats: GenerationStats) -> None:
        self.history.append(stats)
        gen = stats.generation
        num_gen = self.spin_num_gen.value()

        # Actualizar gráficas
        self.plot_function.update_generation(stats)
        self.plot_convergence.update_generation(stats)

        # Actualizar telemetría
        self.progress_bar.setValue(gen)
        self.progress_bar.setFormat(f"Gen {gen} / {num_gen}")

        best_val = stats.best_so_far_raw
        self.card_best.lbl_val.setText(f"{best_val:.6f}")

        opt_val = self.current_func.get_optimum_val(self.spin_n_vars.value())
        err = abs(best_val - opt_val)
        self.card_error.lbl_val.setText(f"{err:.6e}")

        # Indicador de éxito (Verde si el error es menor que la tolerancia)
        try:
            tol = float(self.edit_tol.text())
        except ValueError:
            tol = 0.001

        threshold = max(tol * 10.0, 0.01)
        if err <= threshold:
            self.lbl_status_badge.setText(f"✓ ÓPTIMO ALCANZADO (err < {threshold:.3f})")
            self.lbl_status_badge.setStyleSheet("background-color: #059669; color: #ffffff; padding: 6px; border-radius: 6px; font-weight: bold;")
        else:
            self.lbl_status_badge.setText(f"EN EVOLUCIÓN (err = {err:.4f})")
            self.lbl_status_badge.setStyleSheet("background-color: #0284c7; color: #ffffff; padding: 6px; border-radius: 6px; font-weight: bold;")

        # Coordenadas de best_x
        best_x_str = ", ".join(f"{v:.5f}" for v in stats.best_x)
        self.txt_best_x.setText(f"[{best_x_str}]")

        # Tiempo
        elapsed = time.perf_counter() - self.start_time
        seed = self.edit_seed.text()
        self.lbl_time_info.setText(f"Tiempo: {elapsed:.1f} s | Gen: {gen} | Semilla: {seed}")

        # Tabla de historial reciente (últimas 15 filas)
        if gen % max(1, num_gen // 100) == 0 or gen == num_gen:
            row_pos = self.table_history.rowCount()
            self.table_history.insertRow(row_pos)
            self.table_history.setItem(row_pos, 0, QtWidgets.QTableWidgetItem(str(gen)))
            self.table_history.setItem(row_pos, 1, QtWidgets.QTableWidgetItem(f"{stats.best_fitness_raw:.5f}"))
            self.table_history.setItem(row_pos, 2, QtWidgets.QTableWidgetItem(f"{stats.mean_fitness_raw:.5f}"))
            self.table_history.setItem(row_pos, 3, QtWidgets.QTableWidgetItem(f"{err:.5f}"))
            self.table_history.scrollToBottom()

    def _on_evolution_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_start.setText("▶ Reiniciar / Iniciar")
        self.btn_pause.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Evolución completada con éxito.")

    def _on_export_png_clicked(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Guardar gráfica como PNG",
            f"grafica_{self.current_func.id}_{int(time.time())}.png",
            "Imágenes PNG (*.png)",
        )
        if file_path:
            # Capturar ambas gráficas o la ventana
            pixmap = self.grab()
            pixmap.save(file_path, "PNG")
            self.status_bar.showMessage(f"Gráfica guardada en: {file_path}")

    def _on_export_csv_clicked(self) -> None:
        if not self.history:
            QtWidgets.QMessageBox.information(self, "Sin datos", "No hay historial generado para exportar.")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Guardar historial como CSV",
            f"historial_{self.current_func.id}_{int(time.time())}.csv",
            "Archivos CSV (*.csv)",
        )
        if file_path:
            export_history_to_csv(self.history, file_path)
            self.status_bar.showMessage(f"Historial CSV guardado en: {file_path}")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Algoritmos Genéticos - Proyecto 1")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
