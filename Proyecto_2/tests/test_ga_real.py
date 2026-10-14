"""Pruebas unitarias para el motor del Algoritmo Genético real (Proyecto 2)."""

import numpy as np
import pytest

from common.functions import get_function
from Proyecto_2.ga_real import RealGAConfig, RealGeneticAlgorithm


def test_elitism_monotonicity_real():
    """El elitismo asegura que el mejor histórico nunca empeore en representación real."""
    # Maximización
    f_max = get_function("M2")
    cfg_max = RealGAConfig(function=f_max, tam_pob=50, num_gen=25, elitismo=True, seed=42)
    history_max = [s.best_so_far_raw for s in RealGeneticAlgorithm(cfg_max).run()]
    assert all(history_max[i] <= history_max[i + 1] + 1e-12 for i in range(len(history_max) - 1))

    # Minimización
    f_min = get_function("m1")
    cfg_min = RealGAConfig(function=f_min, tam_pob=50, num_gen=25, elitismo=True, seed=42)
    history_min = [s.best_so_far_raw for s in RealGeneticAlgorithm(cfg_min).run()]
    assert all(history_min[i] >= history_min[i + 1] - 1e-12 for i in range(len(history_min) - 1))


def test_reproducibility_real():
    """Dos ejecuciones con la misma semilla dan resultados idénticos paso a paso."""
    f = get_function("V2")
    cfg1 = RealGAConfig(function=f, n_vars=3, tam_pob=40, num_gen=10, seed=777)
    cfg2 = RealGAConfig(function=f, n_vars=3, tam_pob=40, num_gen=10, seed=777)

    hist1 = list(RealGeneticAlgorithm(cfg1).run())
    hist2 = list(RealGeneticAlgorithm(cfg2).run())

    for s1, s2 in zip(hist1, hist2):
        assert np.isclose(s1.best_fitness_raw, s2.best_fitness_raw, atol=1e-12)
        assert np.allclose(s1.best_x, s2.best_x, atol=1e-12)


def test_all_crossover_operators_run_successfully():
    """Verifica que los 4 operadores de cruce evolucionan sin excepciones."""
    f = get_function("V2")
    operators = ["one_point", "discrete_uniform", "arithmetic", "sbx"]

    for op in operators:
        cfg = RealGAConfig(function=f, n_vars=3, tam_pob=30, num_gen=5, crossover_type=op, seed=123)
        history = list(RealGeneticAlgorithm(cfg).run())
        assert len(history) == 6  # Gen 0 + 5


def test_convergence_m2_real():
    """M2 en real debe converger cerca de 22.3136."""
    f = get_function("M2")
    cfg = RealGAConfig(function=f, tam_pob=100, num_gen=80, crossover_type="sbx", seed=42)
    history = list(RealGeneticAlgorithm(cfg).run())
    assert history[-1].best_so_far_raw >= 21.8


def test_convergence_v4_real():
    """V4 (Schwefel n=3) con óptimo -3*63.635 = -190.9."""
    f = get_function("V4")
    n = 3
    cfg = RealGAConfig(function=f, n_vars=n, tam_pob=100, num_gen=100, crossover_type="sbx", seed=42)
    history = list(RealGeneticAlgorithm(cfg).run())
    final_best = history[-1].best_so_far_raw
    # Debe alcanzar valores muy cercanos al mínimo
    assert final_best <= -150.0
