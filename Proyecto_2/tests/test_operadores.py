"""Pruebas unitarias para los operadores genéticos reales (Proyecto 2)."""

import numpy as np
import pytest

from Proyecto_2.operadores import (
    crossover_arithmetic,
    crossover_discrete_uniform,
    crossover_one_point,
    crossover_sbx,
    mutate_uniform_population,
)


def test_crossover_bounds_clipping():
    """Verifica que todos los operadores devuelven genes estrictamente dentro de los límites."""
    rng = np.random.default_rng(42)
    n_vars = 6
    bounds = np.tile(np.array([-5.0, 5.0]), (n_vars, 1))
    p1 = rng.uniform(-5.0, 5.0, size=n_vars)
    p2 = rng.uniform(-5.0, 5.0, size=n_vars)

    operators = [
        crossover_one_point,
        crossover_discrete_uniform,
        crossover_arithmetic,
        crossover_sbx,
    ]

    for op in operators:
        h1, h2 = op(p1, p2, bounds, rng)
        assert len(h1) == n_vars
        assert len(h2) == n_vars
        assert np.all(h1 >= bounds[:, 0] - 1e-12)
        assert np.all(h1 <= bounds[:, 1] + 1e-12)
        assert np.all(h2 >= bounds[:, 0] - 1e-12)
        assert np.all(h2 <= bounds[:, 1] + 1e-12)


def test_one_point_crossover_n1_edge_case():
    """Con n=1 (1 variable), el cruce de un punto no lanza error y devuelve los padres intactos."""
    rng = np.random.default_rng(42)
    bounds = np.array([[0.0, 10.0]])
    p1 = np.array([3.14])
    p2 = np.array([7.89])

    h1, h2 = crossover_one_point(p1, p2, bounds, rng)
    assert np.allclose(h1, p1)
    assert np.allclose(h2, p2)


def test_arithmetic_crossover_alpha():
    """Aritmético con alpha=0.5 produce dos hijos idénticos (la media de los padres)."""
    rng = np.random.default_rng(42)
    bounds = np.array([[-10.0, 10.0], [-10.0, 10.0]])
    p1 = np.array([2.0, 8.0])
    p2 = np.array([4.0, 2.0])

    # Con alpha = 0.5: h1 = h2 = (p1 + p2)/2 = [3.0, 5.0]
    h1, h2 = crossover_arithmetic(p1, p2, bounds, rng, alpha=0.5)
    expected = np.array([3.0, 5.0])
    assert np.allclose(h1, expected)
    assert np.allclose(h2, expected)

    # Con alpha = 1.0: h1 = p1, h2 = p2
    h1_1, h2_1 = crossover_arithmetic(p1, p2, bounds, rng, alpha=1.0)
    assert np.allclose(h1_1, p1)
    assert np.allclose(h2_1, p2)


def test_sbx_crossover_properties():
    """En SBX, la media de los hijos coincide con la media de los padres."""
    rng = np.random.default_rng(123)
    n_vars = 4
    bounds = np.tile(np.array([-50.0, 50.0]), (n_vars, 1))
    p1 = np.array([-10.0, 5.0, 20.0, -2.0])
    p2 = np.array([10.0, 15.0, 24.0, 8.0])

    h1, h2 = crossover_sbx(p1, p2, bounds, rng, n_sbx=1.0)
    # Media de padres == Media de hijos (salvo clipping en los extremos lejanos)
    mean_parents = 0.5 * (p1 + p2)
    mean_children = 0.5 * (h1 + h2)
    assert np.allclose(mean_parents, mean_children, atol=1e-10)


def test_uniform_mutation():
    """Mutación uniforme con prob_mut=0 no altera la población; con 1.0 cambia todos los no élite."""
    rng = np.random.default_rng(42)
    tam_pob = 10
    n_vars = 3
    bounds = np.tile(np.array([0.0, 10.0]), (n_vars, 1))

    pop = np.full((tam_pob, n_vars), 5.0)
    non_elite_start = 2

    # prob_mut = 0.0 -> nada cambia
    pop_copy = pop.copy()
    mutate_uniform_population(pop_copy, bounds, prob_mut=0.0, non_elite_start=non_elite_start, rng=rng)
    assert np.allclose(pop_copy, pop)

    # prob_mut = 1.0 -> cambian todos los no élite, élite intacta
    pop_mut = pop.copy()
    mutate_uniform_population(pop_mut, bounds, prob_mut=1.0, non_elite_start=non_elite_start, rng=rng)
    # Élite intacta
    assert np.allclose(pop_mut[:non_elite_start], 5.0)
    # No élite modificada
    assert not np.any(np.isclose(pop_mut[non_elite_start:], 5.0))
