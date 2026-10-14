"""Pruebas unitarias para el motor de Algoritmo Genético binario (Proyecto 1)."""

import numpy as np
import pytest

from common.functions import get_function
from common.stats import GenerationStats
from Proyecto_1.ga_binario import BinaryGAConfig, BinaryGeneticAlgorithm


def test_lcrom_calculation():
    # Para intervalo [0, 32] con TOL = 0.0001:
    # 1 + 32/0.0001 = 320001 -> ceil(log2(320001)) = 19
    f = get_function("M2")
    config = BinaryGAConfig(function=f, tol=1e-4, tam_pob=50, num_gen=10)
    ga = BinaryGeneticAlgorithm(config)
    assert ga.lcrom == [19]
    assert ga.total_bits == 19


def test_binary_decoding_boundaries():
    f = get_function("M2")  # [0, 32]
    config = BinaryGAConfig(function=f, tol=1e-4, tam_pob=2, num_gen=1)
    ga = BinaryGeneticAlgorithm(config)

    # 1. Todo ceros -> x = 0.0
    zero_chrom = np.zeros((1, ga.total_bits), dtype=np.uint8)
    x_zero = ga.decode_population(zero_chrom)
    assert np.isclose(x_zero[0, 0], 0.0, atol=1e-12)

    # 2. Todo unos -> x = 32.0
    ones_chrom = np.ones((1, ga.total_bits), dtype=np.uint8)
    x_ones = ga.decode_population(ones_chrom)
    assert np.isclose(x_ones[0, 0], 32.0, atol=1e-12)


def test_least_significant_bit_order():
    """Verifica que el gen 0 es el menos significativo (LSB), idéntico al C++."""
    f = get_function("M1")  # [0, 1]
    config = BinaryGAConfig(function=f, tol=1e-2, tam_pob=2, num_gen=1)
    ga = BinaryGeneticAlgorithm(config)
    lcrom = ga.lcrom[0]

    # gen 0 = 1, resto 0 -> valor binario = 1
    chrom_lsb = np.zeros((1, lcrom), dtype=np.uint8)
    chrom_lsb[0, 0] = 1
    x_lsb = ga.decode_population(chrom_lsb)[0, 0]

    # gen 1 = 1, resto 0 -> valor binario = 2
    chrom_bit1 = np.zeros((1, lcrom), dtype=np.uint8)
    chrom_bit1[0, 1] = 1
    x_bit1 = ga.decode_population(chrom_bit1)[0, 0]

    assert x_lsb < x_bit1
    assert np.isclose(x_bit1, 2.0 * x_lsb, atol=1e-6)


def test_minimization_transformation_negative_cmax():
    """Verifica que con valores negativos (cmax < 0) las aptitudes son positivas y preservan orden."""
    f = get_function("m2")  # Valores negativos alrededor de -200
    config = BinaryGAConfig(function=f, tol=1e-3, tam_pob=10, num_gen=5)
    ga = BinaryGeneticAlgorithm(config)

    raw_neg = np.array([-200.0, -150.0, -100.0, -50.0], dtype=np.float64)
    adapted = ga._transform_fitness_for_minimization(raw_neg)

    # Todas deben ser > 0
    assert np.all(adapted > 0)
    # En minimización, -200 (el mejor) debe tener la MAYOR aptitud adaptada
    assert adapted[0] > adapted[1] > adapted[2] > adapted[3]


def test_elitism_monotonicity():
    """El elitismo garantiza que el mejor histórico (best_so_far_raw) sea monótono."""
    # Test en maximización
    f_max = get_function("M1")
    cfg_max = BinaryGAConfig(function=f_max, tol=1e-3, tam_pob=50, num_gen=20, elitismo=True, seed=123)
    ga_max = BinaryGeneticAlgorithm(cfg_max)
    best_history_max = [s.best_so_far_raw for s in ga_max.run()]
    assert all(best_history_max[i] <= best_history_max[i + 1] + 1e-12 for i in range(len(best_history_max) - 1))

    # Test en minimización
    f_min = get_function("m1")
    cfg_min = BinaryGAConfig(function=f_min, tol=1e-3, tam_pob=50, num_gen=20, elitismo=True, seed=123)
    ga_min = BinaryGeneticAlgorithm(cfg_min)
    best_history_min = [s.best_so_far_raw for s in ga_min.run()]
    assert all(best_history_min[i] >= best_history_min[i + 1] - 1e-12 for i in range(len(best_history_min) - 1))


def test_reproducibility():
    """Dos ejecuciones con la misma semilla producen exactamente los mismos resultados."""
    f = get_function("M2")
    cfg1 = BinaryGAConfig(function=f, tol=1e-3, tam_pob=60, num_gen=15, seed=999)
    cfg2 = BinaryGAConfig(function=f, tol=1e-3, tam_pob=60, num_gen=15, seed=999)

    hist1 = list(BinaryGeneticAlgorithm(cfg1).run())
    hist2 = list(BinaryGeneticAlgorithm(cfg2).run())

    assert len(hist1) == len(hist2) == 16  # Gen 0 + 15 gens
    for s1, s2 in zip(hist1, hist2):
        assert np.isclose(s1.best_fitness_raw, s2.best_fitness_raw, atol=1e-10)
        assert np.isclose(s1.mean_fitness_raw, s2.mean_fitness_raw, atol=1e-10)
        assert np.allclose(s1.best_x, s2.best_x, atol=1e-10)


def test_convergence_m1():
    """M1 debe converger cerca de 1.0 (en x=0.1)."""
    f = get_function("M1")
    cfg = BinaryGAConfig(function=f, tol=1e-4, tam_pob=150, num_gen=60, elitismo=True, seed=42)
    history = list(BinaryGeneticAlgorithm(cfg).run())
    final_best = history[-1].best_so_far_raw
    assert final_best >= 0.95


def test_convergence_m2():
    """M2 debe converger cerca de 22.3136 (en x=31.5005)."""
    f = get_function("M2")
    cfg = BinaryGAConfig(function=f, tol=1e-4, tam_pob=200, num_gen=80, elitismo=True, seed=42)
    history = list(BinaryGeneticAlgorithm(cfg).run())
    final_best = history[-1].best_so_far_raw
    assert final_best >= 21.5


def test_convergence_multivariable_v1():
    """V1 (2 variables) debe converger cerca del óptimo 1.0 en (0, 0)."""
    f = get_function("V1")
    cfg = BinaryGAConfig(function=f, n_vars=2, tol=1e-3, tam_pob=150, num_gen=60, elitismo=True, seed=42)
    history = list(BinaryGeneticAlgorithm(cfg).run())
    final_best = history[-1].best_so_far_raw
    assert final_best >= 0.90
