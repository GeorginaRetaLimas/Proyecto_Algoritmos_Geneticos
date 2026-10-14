"""Aplicación principal con interfaz gráfica para el Proyecto 2 (AG con representación real).

Incluye selector interactivo entre los 4 operadores de cruce (Un punto, Discreto Uniforme,
Aritmético, SBX), modo especial de 'Comparar Operadores', aviso visible en caso de 1 variable
para cruce monopunto, visualización en vivo (PyQtGraph) y control de animación.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from common.functions import ALL_FUNCTIONS, TargetFunction, get_function
from common.stats import GenerationStats, export_history_to_csv
from common.ui.plots import ConvergencePlotWidget, FunctionPopulationPlotWidget
from common.ui.theme import get_theme_stylesheet
from common.ui.worker import GeneticAlgorithmWorker
from Proyecto_2.ga_real import RealGAConfig, RealGeneticAlgorithm
from Proyecto_2.operadores import CROSSOVER_NAMES, CrossoverType


class MainWindowReal(QtWidgets.QMainWindow):
    """Ventana principal del Proyecto 2: AG con representación real."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Algoritmos Genéticos — Proyecto 2: Optimización con Representación Real")
        self.resize(1420, 920)
        self.setMinimumSize(1100, 750)

        # Estado interno
        self.is_dark_theme = True
        self.worker: GeneticAlgorithmWorker | None = None
        self.current_func: TargetFunction = get_function("V2")
        self.history: list[GenerationStats] = []
        self.start_time: float = 0.0

        # Hoja de estilos
        self.setStyleSheet(get_theme_stylesheet(dark=True))

        # Construir UI
        self._init_ui()

        # Cargar valores de la función inicial
        self._on_function_changed("V2")

    def _init_ui(self) -> None:
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)

        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # 1. PANEL IZQUIERDO: Configuración y controles (Sidebar)
        sidebar_scroll = QtWidgets.QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(390)
        sidebar_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        sidebar_content = QtWidgets.QWidget()
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(0, 0, 8, 0)
        sidebar_layout.setSpacing(12)

        sidebar_layout.addWidget(self._create_function_group())
        sidebar_layout.addWidget(self._create_crossover_operator_group())
        sidebar_layout.addWidget(self._create_ga_params_group())
        sidebar_layout.addWidget(self._create_animation_control_group())
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

        # 3. PANEL DERECHO: Resultados y Comparativa de Operadores
        right_panel = self._create_results_panel()
        main_layout.addWidget(right_panel, stretch=1)

        # Barra de estado
        self.status_bar = QtWidgets.QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Listo. Seleccione función y operador de cruce.")

    # -----------------------------------------------------------------------
    # Grupos de Configuración
    # -----------------------------------------------------------------------

    def _create_function_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("1. Función Objetivo")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        # Selector de función
        func_layout = QtWidgets.QHBoxLayout()
        func_label = QtWidgets.QLabel("Función:")
        self.combo_function = QtWidgets.QComboBox()
        for f in ALL_FUNCTIONS:
            self.combo_function.addItem(f"{f.id} — {f.name}", userData=f.id)
        # Seleccionar V2 por defecto (Rastrigin multivariable)
        idx_v2 = [i for i, f in enumerate(ALL_FUNCTIONS) if f.id == "V2"]
        if idx_v2:
            self.combo_function.setCurrentIndex(idx_v2[0])
        self.combo_function.currentIndexChanged.connect(self._on_combo_function_selected)
        func_layout.addWidget(func_label)
        func_layout.addWidget(self.combo_function)
        layout.addLayout(func_layout)

        # Tarjeta informativa
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

    def _create_crossover_operator_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("2. Operador de Cruce Real")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        # Combo selector de operador
        op_layout = QtWidgets.QHBoxLayout()
        op_layout.addWidget(QtWidgets.QLabel("Operador:"))
        self.combo_crossover = QtWidgets.QComboBox()
        for key, name in CROSSOVER_NAMES.items():
            self.combo_crossover.addItem(name, userData=key)
        self.combo_crossover.currentIndexChanged.connect(self._on_crossover_operator_changed)
        op_layout.addWidget(self.combo_crossover)
        layout.addLayout(op_layout)

        # Parámetro SBX: n_sbx
        self.frame_sbx = QtWidgets.QFrame()
        layout_sbx = QtWidgets.QHBoxLayout(self.frame_sbx)
        layout_sbx.setContentsMargins(0, 0, 0, 0)
        layout_sbx.addWidget(QtWidgets.QLabel("Índice n_sbx:"))
        self.spin_n_sbx = QtWidgets.QDoubleSpinBox()
        self.spin_n_sbx.setRange(0.1, 10.0)
        self.spin_n_sbx.setValue(1.0)
        self.spin_n_sbx.setSingleStep(0.5)
        layout_sbx.addWidget(self.spin_n_sbx)
        layout.addWidget(self.frame_sbx)

        # Parámetro Aritmético: alpha
        self.frame_arithmetic = QtWidgets.QFrame()
        layout_arith = QtWidgets.QHBoxLayout(self.frame_arithmetic)
        layout_arith.setContentsMargins(0, 0, 0, 0)
        layout_arith.addWidget(QtWidgets.QLabel("Factor α (0..1):"))
        self.spin_alpha = QtWidgets.QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setValue(0.5)
        self.spin_alpha.setSingleStep(0.05)
        layout_arith.addWidget(self.spin_alpha)
        self.chk_rand_alpha = QtWidgets.QCheckBox("Aleatorio")
        layout_arith.addWidget(self.chk_rand_alpha)
        layout.addWidget(self.frame_arithmetic)

        # Parámetro Uniforme: p_u
        self.frame_uniform = QtWidgets.QFrame()
        layout_u = QtWidgets.QHBoxLayout(self.frame_uniform)
        layout_u.setContentsMargins(0, 0, 0, 0)
        layout_u.addWidget(QtWidgets.QLabel("Prob. intercambio p_u:"))
        self.spin_pu = QtWidgets.QDoubleSpinBox()
        self.spin_pu.setRange(0.01, 0.99)
        self.spin_pu.setValue(0.5)
        self.spin_pu.setSingleStep(0.05)
        layout_u.addWidget(self.spin_pu)
        layout.addWidget(self.frame_uniform)

        # Aviso visual para 1 variable + Cruce de un punto
        self.lbl_warning_one_point = QtWidgets.QLabel("⚠️ Con 1 variable, el cruce de un punto no altera a los padres.")
        self.lbl_warning_one_point.setStyleSheet("color: #fbbf24; background: #78350f55; border-radius: 4px; padding: 4px; font-size: 11px;")
        self.lbl_warning_one_point.setWordWrap(True)
        self.lbl_warning_one_point.setVisible(False)
        layout.addWidget(self.lbl_warning_one_point)

        self._on_crossover_operator_changed(0)
        return group

    def _create_ga_params_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("3. Parámetros del AG Real")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        form_layout = QtWidgets.QGridLayout()
        form_layout.setSpacing(8)

        # Variables n
        form_layout.addWidget(QtWidgets.QLabel("Variables (n):"), 0, 0)
        self.spin_n_vars = QtWidgets.QSpinBox()
        self.spin_n_vars.setRange(1, 30)
        self.spin_n_vars.setValue(5)
        self.spin_n_vars.valueChanged.connect(self._on_n_vars_changed)
        form_layout.addWidget(self.spin_n_vars, 0, 1)

        # Dominio [inf, sup]
        form_layout.addWidget(QtWidgets.QLabel("Límites [inf, sup]:"), 1, 0)
        bounds_layout = QtWidgets.QHBoxLayout()
        self.spin_xmin = QtWidgets.QDoubleSpinBox()
        self.spin_xmin.setRange(-100000.0, 100000.0)
        self.spin_xmin.setDecimals(4)

        self.spin_xmax = QtWidgets.QDoubleSpinBox()
        self.spin_xmax.setRange(-100000.0, 100000.0)
        self.spin_xmax.setDecimals(4)

        bounds_layout.addWidget(self.spin_xmin)
        bounds_layout.addWidget(QtWidgets.QLabel("a"))
        bounds_layout.addWidget(self.spin_xmax)
        form_layout.addLayout(bounds_layout, 1, 1)

        # Población
        form_layout.addWidget(QtWidgets.QLabel("Población:"), 2, 0)
        self.spin_tam_pob = QtWidgets.QSpinBox()
        self.spin_tam_pob.setRange(10, 2000)
        self.spin_tam_pob.setValue(100)
        self.spin_tam_pob.setSingleStep(20)
        form_layout.addWidget(self.spin_tam_pob, 2, 1)

        # Generaciones
        form_layout.addWidget(QtWidgets.QLabel("Generaciones:"), 3, 0)
        self.spin_num_gen = QtWidgets.QSpinBox()
        self.spin_num_gen.setRange(10, 20000)
        self.spin_num_gen.setValue(500)
        self.spin_num_gen.setSingleStep(50)
        self.spin_num_gen.valueChanged.connect(self._update_delay_from_duration)
        form_layout.addWidget(self.spin_num_gen, 3, 1)

        # Prob Cruce
        form_layout.addWidget(QtWidgets.QLabel("Prob. Cruce:"), 4, 0)
        self.spin_prob_cruce = QtWidgets.QDoubleSpinBox()
        self.spin_prob_cruce.setRange(0.0, 1.0)
        self.spin_prob_cruce.setValue(0.80)
        self.spin_prob_cruce.setSingleStep(0.05)
        form_layout.addWidget(self.spin_prob_cruce, 4, 1)

        # Prob Mutación (por gen)
        form_layout.addWidget(QtWidgets.QLabel("Prob. Mutación (gen):"), 5, 0)
        self.spin_prob_mut = QtWidgets.QDoubleSpinBox()
        self.spin_prob_mut.setRange(0.0, 1.0)
        self.spin_prob_mut.setValue(0.05)
        self.spin_prob_mut.setSingleStep(0.01)
        self.spin_prob_mut.setDecimals(3)
        form_layout.addWidget(self.spin_prob_mut, 5, 1)

        # Semilla
        form_layout.addWidget(QtWidgets.QLabel("Semilla RNG:"), 6, 0)
        seed_layout = QtWidgets.QHBoxLayout()
        self.edit_seed = QtWidgets.QLineEdit("")
        self.edit_seed.setPlaceholderText("Aleatoria")
        self.btn_new_seed = QtWidgets.QPushButton("🎲")
        self.btn_new_seed.setFixedWidth(36)
        self.btn_new_seed.clicked.connect(self._generate_random_seed)
        seed_layout.addWidget(self.edit_seed)
        seed_layout.addWidget(self.btn_new_seed)
        form_layout.addLayout(seed_layout, 6, 1)

        layout.addLayout(form_layout)

        # Elitismo
        elite_layout = QtWidgets.QHBoxLayout()
        self.chk_elitism = QtWidgets.QCheckBox("Elitismo activado")
        self.chk_elitism.setChecked(True)
        self.spin_pct_elite = QtWidgets.QDoubleSpinBox()
        self.spin_pct_elite.setRange(0.1, 50.0)
        self.spin_pct_elite.setValue(2.0)
        self.spin_pct_elite.setSuffix(" %")
        elite_layout.addWidget(self.chk_elitism)
        elite_layout.addWidget(self.spin_pct_elite)
        layout.addLayout(elite_layout)

        return group

    def _create_animation_control_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("4. Velocidad de Animación")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

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

        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("Retardo:"))
        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_speed.setRange(0, 1000)
        self.slider_speed.setValue(20)
        self.slider_speed.valueChanged.connect(self._on_speed_slider_changed)
        slider_layout.addWidget(self.slider_speed)

        self.lbl_speed_val = QtWidgets.QLabel("20 ms/gen")
        self.lbl_speed_val.setFixedWidth(75)
        slider_layout.addWidget(self.lbl_speed_val)
        layout.addLayout(slider_layout)

        self.lbl_duration_calc = QtWidgets.QLabel("Duración estimada: ~10 s")
        self.lbl_duration_calc.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_duration_calc)

        return group

    def _create_actions_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("5. Ejecución")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(8)

        # Botón Iniciar
        self.btn_start = QtWidgets.QPushButton("▶ Iniciar Evolución")
        self.btn_start.setObjectName("btnPrimary")
        self.btn_start.setFixedHeight(36)
        self.btn_start.clicked.connect(self._on_start_clicked)
        layout.addWidget(self.btn_start)

        # Botón Comparar 4 Operadores
        self.btn_compare_all = QtWidgets.QPushButton("⚡ Comparar 4 Operadores")
        self.btn_compare_all.setObjectName("btnSuccess")
        self.btn_compare_all.setFixedHeight(32)
        self.btn_compare_all.clicked.connect(self._on_compare_all_clicked)
        layout.addWidget(self.btn_compare_all)

        # Controles Pausa/Paso/Detener
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

        # Reiniciar y exportar
        self.btn_reset = QtWidgets.QPushButton("🔄 Reiniciar")
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        layout.addWidget(self.btn_reset)

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

        # Encabezado
        header_layout = QtWidgets.QHBoxLayout()
        title_lbl = QtWidgets.QLabel("Telemetría Real")
        title_lbl.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        self.btn_toggle_theme = QtWidgets.QPushButton("🌓 Tema")
        self.btn_toggle_theme.setFixedWidth(70)
        self.btn_toggle_theme.clicked.connect(self._toggle_theme)
        header_layout.addWidget(title_lbl)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_toggle_theme)
        layout.addLayout(header_layout)

        # Barra de progreso
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 500)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Gen 0 / 500")
        layout.addWidget(self.progress_bar)

        # Tarjetas KPI
        self.card_best = self._create_kpi_card("Mejor Aptitud Real f(x)", "--", color="#10b981")
        layout.addWidget(self.card_best)

        self.card_error = self._create_kpi_card("Error (|f - f*|)", "--", color="#38bdf8")
        layout.addWidget(self.card_error)

        self.lbl_status_badge = QtWidgets.QLabel("LISTO")
        self.lbl_status_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_status_badge.setStyleSheet("background-color: #334155; color: #ffffff; padding: 6px; border-radius: 6px; font-weight: bold;")
        layout.addWidget(self.lbl_status_badge)

        self.lbl_best_x_title = QtWidgets.QLabel("Mejor Cromosoma Real:")
        self.lbl_best_x_title.setStyleSheet("font-weight: 600; color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_best_x_title)

        self.txt_best_x = QtWidgets.QTextEdit()
        self.txt_best_x.setReadOnly(True)
        self.txt_best_x.setFixedHeight(65)
        self.txt_best_x.setPlaceholderText("Valores continuos del mejor individuo...")
        layout.addWidget(self.txt_best_x)

        self.lbl_time_info = QtWidgets.QLabel("Tiempo: 0.0 s | Operador: SBX")
        self.lbl_time_info.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(self.lbl_time_info)

        # Tabla de comparación / historial
        self.lbl_table_title = QtWidgets.QLabel("Resultados de Operadores / Historial:")
        layout.addWidget(self.lbl_table_title)
        self.table_results = QtWidgets.QTableWidget()
        self.table_results.setColumnCount(4)
        self.table_results.setHorizontalHeaderLabels(["Operador", "Mejor f(x)", "Error", "Tiempo"])
        self.table_results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_results.verticalHeader().setVisible(False)
        layout.addWidget(self.table_results)

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
        card.lbl_val = lbl_v
        return card

    # -----------------------------------------------------------------------
    # Lógica de Eventos
    # -----------------------------------------------------------------------

    def _toggle_theme(self) -> None:
        self.is_dark_theme = not self.is_dark_theme
        self.setStyleSheet(get_theme_stylesheet(dark=self.is_dark_theme))

    def _on_crossover_operator_changed(self, index: int) -> None:
        op_key = self.combo_crossover.currentData()
        self.frame_sbx.setVisible(op_key == "sbx")
        self.frame_arithmetic.setVisible(op_key == "arithmetic")
        self.frame_uniform.setVisible(op_key == "discrete_uniform")
        self._check_one_point_warning()

    def _check_one_point_warning(self) -> None:
        if not hasattr(self, "spin_n_vars") or not hasattr(self, "combo_crossover"):
            return
        op_key = self.combo_crossover.currentData()
        n_vars = self.spin_n_vars.value()
        self.lbl_warning_one_point.setVisible(op_key == "one_point" and n_vars == 1)

    def _on_combo_function_selected(self, index: int) -> None:
        if index < 0:
            return
        func_id = self.combo_function.itemData(index)
        if func_id:
            self._on_function_changed(func_id)

    def _on_function_changed(self, func_id: str) -> None:
        self.current_func = get_function(func_id)
        f = self.current_func

        self.lbl_formula_display.setText(f.formula_display)
        target_type_str = "MAXIMIZACIÓN" if f.target_type == "max" else "MINIMIZACIÓN"

        x_min, x_max = f.bounds
        self.spin_xmin.setValue(x_min)
        self.spin_xmax.setValue(x_max)

        self.spin_n_vars.blockSignals(True)
        self.spin_n_vars.setValue(f.default_n_vars)
        self.spin_n_vars.setEnabled(f.is_n_configurable)
        self.spin_n_vars.blockSignals(False)

        n_vars = self.spin_n_vars.value()
        opt_val = f.get_optimum_val(n_vars)
        self.lbl_optimum_info.setText(f"Tipo: {target_type_str} | Óptimo: {opt_val:.4f}")

        bounds_arr = f.get_bounds_array(n_vars)
        self.plot_function.set_function_context(f, n_vars, bounds_arr)
        self.plot_convergence.set_optimum(opt_val)
        self.plot_convergence.reset()

        self._check_one_point_warning()
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
            self._check_one_point_warning()

    def _generate_random_seed(self) -> None:
        self.edit_seed.setText(str(int(np.random.default_rng().integers(1, 999999))))

    def _set_target_duration(self, seconds: float) -> None:
        num_gen = max(1, self.spin_num_gen.value())
        delay_ms = 0 if seconds <= 0.0 else int(round((seconds / num_gen) * 1000.0))
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
            self.lbl_duration_calc.setText("Duración estimada: instantáneo")
        else:
            self.lbl_speed_val.setText(f"{value} ms/gen")
            self.lbl_duration_calc.setText(
                f"Duración estimada: ~{total_est:.1f} s" if total_est < 60 else f"Duración estimada: ~{total_est / 60:.1f} min"
            )

        if self.worker is not None:
            self.worker.set_delay(delay_seconds)

    # -----------------------------------------------------------------------
    # Ejecución Individual y Comparativa
    # -----------------------------------------------------------------------

    def _get_config_from_ui(self, override_crossover: CrossoverType | None = None) -> RealGAConfig:
        x_min = self.spin_xmin.value()
        x_max = self.spin_xmax.value()
        n_vars = self.spin_n_vars.value()
        bounds_arr = np.tile(np.array([x_min, x_max], dtype=np.float64), (n_vars, 1))

        seed_text = self.edit_seed.text().strip()
        seed_val = int(seed_text) if seed_text.isdigit() else None

        c_type = override_crossover or self.combo_crossover.currentData()

        return RealGAConfig(
            function=self.current_func,
            n_vars=n_vars,
            bounds=bounds_arr,
            tam_pob=self.spin_tam_pob.value(),
            num_gen=self.spin_num_gen.value(),
            prob_cruce=self.spin_prob_cruce.value(),
            prob_mut=self.spin_prob_mut.value(),
            crossover_type=c_type,
            p_u=self.spin_pu.value(),
            alpha=None if self.chk_rand_alpha.isChecked() else self.spin_alpha.value(),
            random_alpha_per_gene=self.chk_rand_alpha.isChecked(),
            n_sbx=self.spin_n_sbx.value(),
            elitismo=self.chk_elitism.isChecked(),
            pct_elite=self.spin_pct_elite.value(),
            seed=seed_val,
        )

    def _on_start_clicked(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.resume()
            self.btn_start.setText("▶ Reanudado")
            self.btn_pause.setEnabled(True)
            self.btn_step.setEnabled(True)
            self.btn_stop.setEnabled(True)
            return

        config = self._get_config_from_ui()
        ga = RealGeneticAlgorithm(config)
        self.edit_seed.setText(str(ga.used_seed))

        self.history.clear()
        self.table_results.setRowCount(0)
        self.table_results.setHorizontalHeaderLabels(["Gen", "Mejor f(x)", "Error", "Operador"])
        self.progress_bar.setRange(0, config.num_gen)
        self.progress_bar.setValue(0)
        self.plot_convergence.reset()
        self.plot_function.set_function_context(self.current_func, config.n_vars, config.bounds)

        delay_seconds = self.slider_speed.value() / 1000.0

        self.worker = GeneticAlgorithmWorker(ga.run(), delay_seconds=delay_seconds)
        self.worker.generation_updated.connect(self._on_generation_updated)
        self.worker.evolution_finished.connect(self._on_evolution_finished)
        self.worker.status_changed.connect(lambda msg: self.status_bar.showMessage(msg))
        self.worker.error_occurred.connect(lambda err: QtWidgets.QMessageBox.critical(self, "Error", err))

        self.start_time = time.perf_counter()
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_step.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.status_bar.showMessage("Evolución real en curso...")

    def _on_compare_all_clicked(self) -> None:
        """Ejecuta los 4 operadores de cruce sobre la misma población/semilla y compara sus curvas."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()

        self.status_bar.showMessage("Ejecutando comparativa de los 4 operadores...")
        base_seed = int(self.edit_seed.text()) if self.edit_seed.text().isdigit() else 42
        self.edit_seed.setText(str(base_seed))

        operators: list[CrossoverType] = ["sbx", "arithmetic", "discrete_uniform", "one_point"]
        colors = {"sbx": "#38bdf8", "arithmetic": "#10b981", "discrete_uniform": "#f59e0b", "one_point": "#ec4899"}

        self.plot_convergence.reset()
        self.table_results.setRowCount(0)
        self.table_results.setHorizontalHeaderLabels(["Operador", "Mejor Final", "Error Final", "Tiempo"])

        opt_val = self.current_func.get_optimum_val(self.spin_n_vars.value())
        self.plot_convergence.set_optimum(opt_val)

        for op in operators:
            cfg = self._get_config_from_ui(override_crossover=op)
            cfg.seed = base_seed

            t0 = time.perf_counter()
            ga = RealGeneticAlgorithm(cfg)
            hist = list(ga.run())
            dt = time.perf_counter() - t0

            # Trazar curva de este operador en la gráfica de convergencia
            gens = [s.generation for s in hist]
            bests = [s.best_so_far_raw for s in hist]
            op_label = CROSSOVER_NAMES[op]
            self.plot_convergence.plot(gens, bests, pen=pg.mkPen(color=colors[op], width=2.0), name=op_label)

            final_best = hist[-1].best_so_far_raw
            final_err = abs(final_best - opt_val)

            # Insertar en tabla
            row = self.table_results.rowCount()
            self.table_results.insertRow(row)
            self.table_results.setItem(row, 0, QtWidgets.QTableWidgetItem(op.upper()))
            self.table_results.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{final_best:.5f}"))
            self.table_results.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{final_err:.5f}"))
            self.table_results.setItem(row, 3, QtWidgets.QTableWidgetItem(f"{dt:.3f} s"))

            # Actualizar última población en Gráfica A
            self.plot_function.update_generation(hist[-1])

        self.status_bar.showMessage("Comparativa de 4 operadores completada.")

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
        self.card_best.lbl_val.setText("--")
        self.card_error.lbl_val.setText("--")
        self.txt_best_x.clear()
        self.table_results.setRowCount(0)
        self.lbl_status_badge.setText("REINICIADO")
        self.lbl_status_badge.setStyleSheet("background-color: #334155; color: #ffffff; padding: 6px; border-radius: 6px; font-weight: bold;")
        self.btn_start.setEnabled(True)
        self.btn_start.setText("▶ Iniciar Evolución")
        self.btn_pause.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Listo para nueva ejecución.")

    def _on_generation_updated(self, stats: GenerationStats) -> None:
        self.history.append(stats)
        gen = stats.generation
        num_gen = self.spin_num_gen.value()

        self.plot_function.update_generation(stats)
        self.plot_convergence.update_generation(stats)

        self.progress_bar.setValue(gen)
        self.progress_bar.setFormat(f"Gen {gen} / {num_gen}")

        best_val = stats.best_so_far_raw
        self.card_best.lbl_val.setText(f"{best_val:.6f}")

        opt_val = self.current_func.get_optimum_val(self.spin_n_vars.value())
        err = abs(best_val - opt_val)
        self.card_error.lbl_val.setText(f"{err:.6e}")

        threshold = 0.05
        if err <= threshold:
            self.lbl_status_badge.setText(f"✓ ÓPTIMO ALCANZADO (err < {threshold})")
            self.lbl_status_badge.setStyleSheet("background-color: #059669; color: #ffffff; padding: 6px; border-radius: 6px; font-weight: bold;")
        else:
            self.lbl_status_badge.setText(f"EN EVOLUCIÓN (err = {err:.4f})")
            self.lbl_status_badge.setStyleSheet("background-color: #0284c7; color: #ffffff; padding: 6px; border-radius: 6px; font-weight: bold;")

        best_x_str = ", ".join(f"{v:.5f}" for v in stats.best_x)
        self.txt_best_x.setText(f"[{best_x_str}]")

        elapsed = time.perf_counter() - self.start_time
        op_name = self.combo_crossover.currentText().split("(")[0].strip()
        self.lbl_time_info.setText(f"Tiempo: {elapsed:.1f} s | Operador: {op_name}")

        if gen % max(1, num_gen // 50) == 0 or gen == num_gen:
            row = self.table_results.rowCount()
            self.table_results.insertRow(row)
            self.table_results.setItem(row, 0, QtWidgets.QTableWidgetItem(str(gen)))
            self.table_results.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{stats.best_fitness_raw:.5f}"))
            self.table_results.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{err:.5f}"))
            self.table_results.setItem(row, 3, QtWidgets.QTableWidgetItem(op_name))
            self.table_results.scrollToBottom()

    def _on_evolution_finished(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_start.setText("▶ Iniciar / Reiniciar")
        self.btn_pause.setEnabled(False)
        self.btn_step.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Evolución completada.")

    def _on_export_png_clicked(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Guardar gráfica como PNG",
            f"grafica_real_{self.current_func.id}_{int(time.time())}.png",
            "Imágenes PNG (*.png)",
        )
        if file_path:
            self.grab().save(file_path, "PNG")
            self.status_bar.showMessage(f"Gráfica guardada en: {file_path}")

    def _on_export_csv_clicked(self) -> None:
        if not self.history:
            QtWidgets.QMessageBox.information(self, "Sin datos", "No hay historial generado para exportar.")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Guardar historial como CSV",
            f"historial_real_{self.current_func.id}_{int(time.time())}.csv",
            "Archivos CSV (*.csv)",
        )
        if file_path:
            export_history_to_csv(self.history, file_path)
            self.status_bar.showMessage(f"Historial CSV guardado en: {file_path}")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Algoritmos Genéticos - Proyecto 2")
    window = MainWindowReal()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
