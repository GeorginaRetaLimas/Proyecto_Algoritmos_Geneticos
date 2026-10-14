"""Pruebas unitarias para el catálogo de funciones objetivo (Proyecto 1 y 2)."""

import numpy as np
import pytest

from common.functions import CATALOG, ALL_FUNCTIONS, TargetFunction, get_function, V3_MINIMA_TABLE


def test_catalog_contains_all_12_functions():
    assert len(CATALOG) == 12
    expected_ids = {"M1", "M2", "M3", "M4", "m1", "m2", "m3", "m4", "V1", "V2", "V3", "V4"}
    assert set(CATALOG.keys()) == expected_ids


def test_m1_optimum():
    f = get_function("M1")
    x_opt = f.get_optimum_x(1)
    val = f.evaluate(x_opt)
    assert np.isclose(val, 1.0, atol=1e-6)


def test_m2_optimum():
    f = get_function("M2")
    x_opt = f.get_optimum_x(1)
    val = f.evaluate(x_opt)
    # Óptimo del libro 22.3136 en x=31.5005
    assert np.isclose(val, 22.3136, atol=1e-3)


def test_m3_optimum():
    f = get_function("M3")
    x_opt = f.get_optimum_x(1)
    val = f.evaluate(x_opt)
    # Óptimo del libro 1.98442447 en x=0.98447395
    assert np.isclose(val, 1.98442447, atol=1e-4)


def test_m4_optimum():
    f = get_function("M4")
    x_opt = f.get_optimum_x(1)
    val = f.evaluate(x_opt)
    assert np.isclose(val, 1.0, atol=1e-6)


def test_min1_optimum():
    f = get_function("m1")
    x_opt = f.get_optimum_x(1)
    val = f.evaluate(x_opt)
    # Óptimo del libro 0.0883634 en x=3.08531
    assert np.isclose(val, 0.0883634, atol=1e-3)


def test_min2_optimum():
    f = get_function("m2")
    x_opt = f.get_optimum_x(1)
    val = f.evaluate(x_opt)
    # Óptimo del libro -201.843 en x=203.814
    assert np.isclose(val, -201.843, atol=1e-2)
    # Comprobar simetría f(-x) == f(x)
    val_neg = f.evaluate(-x_opt)
    assert np.isclose(val_neg, -201.843, atol=1e-2)


def test_min3_optimum():
    f = get_function("m3")
    x_opt = f.get_optimum_x(1)
    val = f.evaluate(x_opt)
    # Óptimo del libro -0.318071 en x=4.5798
    assert np.isclose(val, -0.318071, atol=1e-4)


def test_min4_optimum():
    f = get_function("m4")
    x_opt = f.get_optimum_x(1)
    val = f.evaluate(x_opt)
    # Óptimo del libro con fórmula corregida -5.39204 en x=23.9901
    assert np.isclose(val, -5.39204, atol=1e-3)


def test_v1_optimum():
    f = get_function("V1")
    x_opt = f.get_optimum_x(2)
    val = f.evaluate(x_opt)
    assert np.isclose(val, 1.0, atol=1e-6)


def test_v2_optimum():
    f = get_function("V2")
    for n in (2, 5, 10):
        x_opt = f.get_optimum_x(n)
        val = f.evaluate(x_opt)
        assert np.isclose(val, 0.0, atol=1e-6)


def test_v3_optimum_n1():
    f = get_function("V3")
    # Para n=1, x = pi/2 -> f(x) = -1.0
    x_opt = np.array([np.pi / 2.0])
    val = f.evaluate(x_opt)
    assert np.isclose(val, -1.0, atol=1e-5)


def test_v3_table_consistency():
    f = get_function("V3")
    for n, expected_min in V3_MINIMA_TABLE.items():
        assert f.get_optimum_val(n) == expected_min


def test_v4_optimum():
    f = get_function("V4")
    for n in (2, 5, 10):
        x_opt = f.get_optimum_x(n)
        val = f.evaluate(x_opt)
        expected = f.get_optimum_val(n)
        assert np.isclose(val, expected, atol=1e-2)


def test_vectorized_vs_scalar():
    """Verifica que la evaluación por lote (batch) coincide exactamente con individuales."""
    rng = np.random.default_rng(42)
    for func in ALL_FUNCTIONS:
        n_vars = func.default_n_vars
        bounds = func.bounds
        pop_x = rng.uniform(bounds[0], bounds[1], size=(25, n_vars))
        batch_res = func.evaluate(pop_x)
        assert batch_res.shape == (25,)
        for i in range(25):
            single_res = func.evaluate(pop_x[i])
            assert np.isclose(batch_res[i], single_res, atol=1e-10)
