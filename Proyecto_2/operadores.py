"""Operadores genéticos para representación real (Proyecto 2).

Contiene los 4 operadores de cruce (Un punto, Discreto Uniforme, Aritmético, SBX)
y el operador de mutación uniforme, con reparación/clipping estricto de límites.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

CrossoverType = Literal["one_point", "discrete_uniform", "arithmetic", "sbx"]


def crossover_one_point(
    padre1: np.ndarray,
    padre2: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruce de un punto (discreto) sobre genes reales.

    Si el cromosoma tiene longitud 1 (función de 1 variable), no existe punto de corte
    válido en [1, lcrom - 1], por lo que devuelve copias intactas de los padres.
    """
    n_vars = len(padre1)
    if n_vars <= 1:
        return padre1.copy(), padre2.copy()

    # Punto de corte p en [1, n_vars - 1]
    p = int(rng.integers(1, n_vars))
    hijo1 = np.concatenate([padre1[:p], padre2[p:]])
    hijo2 = np.concatenate([padre2[:p], padre1[p:]])

    # Clipping de seguridad dentro de los límites
    np.clip(hijo1, bounds[:, 0], bounds[:, 1], out=hijo1)
    np.clip(hijo2, bounds[:, 0], bounds[:, 1], out=hijo2)
    return hijo1, hijo2


def crossover_discrete_uniform(
    padre1: np.ndarray,
    padre2: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
    p_u: float = 0.5,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruce discreto uniforme con probabilidad p_u de intercambio gen a gen."""
    n_vars = len(padre1)
    mask = rng.random(n_vars) < p_u

    hijo1 = np.where(mask, padre2, padre1)
    hijo2 = np.where(mask, padre1, padre2)

    np.clip(hijo1, bounds[:, 0], bounds[:, 1], out=hijo1)
    np.clip(hijo2, bounds[:, 0], bounds[:, 1], out=hijo2)
    return hijo1, hijo2


def crossover_arithmetic(
    padre1: np.ndarray,
    padre2: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
    alpha: float | None = 0.5,
    random_alpha_per_gene: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruce aritmético: hijo1 = a*p1 + (1-a)*p2; hijo2 = a*p2 + (1-a)*p1.

    Args:
        alpha: Factor de ponderación en [0, 1]. Si es None o random_alpha_per_gene=True,
               se genera aleatorio U(0, 1).
    """
    n_vars = len(padre1)
    if random_alpha_per_gene:
        a = rng.uniform(0.0, 1.0, size=n_vars)
    elif alpha is None:
        a = float(rng.uniform(0.0, 1.0))
    else:
        a = float(np.clip(alpha, 0.0, 1.0))

    hijo1 = a * padre1 + (1.0 - a) * padre2
    hijo2 = a * padre2 + (1.0 - a) * padre1

    np.clip(hijo1, bounds[:, 0], bounds[:, 1], out=hijo1)
    np.clip(hijo2, bounds[:, 0], bounds[:, 1], out=hijo2)
    return hijo1, hijo2


def crossover_sbx(
    padre1: np.ndarray,
    padre2: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
    n_sbx: float = 1.0,
    alpha_per_gene: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Cruce SBX (Simulated Binary Crossover).

    Acepta el índice de distribución n_sbx (por defecto 1.0 según libro)
    y genera alpha una vez por pareja (por defecto del libro) o por gen.
    """
    n_vars = len(padre1)
    exponent = 1.0 / (max(n_sbx, 0.0) + 1.0)

    if alpha_per_gene:
        u = rng.uniform(1e-7, 1.0 - 1e-7, size=n_vars)
    else:
        u = float(rng.uniform(1e-7, 1.0 - 1e-7))

    # Cálculo del factor de dispersión beta
    if isinstance(u, np.ndarray):
        beta = np.empty_like(u)
        mask = u < 0.5
        beta[mask] = (2.0 * u[mask]) ** exponent
        beta[~mask] = (1.0 / (2.0 * (1.0 - u[~mask]))) ** exponent
    else:
        if u < 0.5:
            beta = (2.0 * u) ** exponent
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** exponent

    diff = np.abs(padre2 - padre1)
    sum_p = padre1 + padre2

    hijo1 = 0.5 * (sum_p - beta * diff)
    hijo2 = 0.5 * (sum_p + beta * diff)

    # Reparación estricta de límites (clipping obligatorio)
    np.clip(hijo1, bounds[:, 0], bounds[:, 1], out=hijo1)
    np.clip(hijo2, bounds[:, 0], bounds[:, 1], out=hijo2)
    return hijo1, hijo2


# Catálogo de operadores de cruce
CROSSOVER_OPERATORS: dict[str, Callable] = {
    "one_point": crossover_one_point,
    "discrete_uniform": crossover_discrete_uniform,
    "arithmetic": crossover_arithmetic,
    "sbx": crossover_sbx,
}

CROSSOVER_NAMES: dict[str, str] = {
    "sbx": "SBX (Simulated Binary Crossover)",
    "arithmetic": "Aritmético",
    "discrete_uniform": "Discreto Uniforme",
    "one_point": "Un Punto (Discreto)",
}


def mutate_uniform_population(
    population: np.ndarray,
    bounds: np.ndarray,
    prob_mut: float,
    non_elite_start: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Aplica mutación uniforme a los individuos no élite de la población.

    Para cada gen con probabilidad prob_mut, se sustituye por un valor aleatorio
    uniforme en el intervalo [inf_j, sup_j] de ese gen.
    """
    tam_pob, n_vars = population.shape
    n_non_elite = tam_pob - non_elite_start
    if n_non_elite <= 0 or prob_mut <= 0.0:
        return population

    # Máscara booleana de mutación
    mask = rng.random((n_non_elite, n_vars)) < prob_mut

    if np.any(mask):
        # Muestreo de valores uniformes por variable dentro de sus límites
        lows = bounds[:, 0]
        highs = bounds[:, 1]
        random_replacements = rng.uniform(lows, highs, size=(n_non_elite, n_vars))

        # Reemplazar únicamente los genes seleccionados
        non_elite_view = population[non_elite_start:]
        non_elite_view[mask] = random_replacements[mask]

    return population
