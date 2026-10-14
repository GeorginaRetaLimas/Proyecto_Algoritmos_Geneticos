"""Catálogo de funciones objetivo para los Proyectos 1 y 2 de Algoritmos Genéticos.

Define las 12 funciones de prueba (4 de maximización 1D, 4 de minimización 1D
y 4 multivariables) con sus dominios, tipos, fórmulas y óptimos teóricos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class TargetFunction:
    """Representa una función objetivo del catálogo."""

    id: str
    name: str
    category: str  # 'max_1d', 'min_1d', 'multi'
    target_type: str  # 'max' o 'min'
    formula_latex: str
    formula_display: str
    default_n_vars: int
    is_n_configurable: bool
    bounds: tuple[float, float]
    # Función que devuelve el óptimo teórico según n_vars
    get_optimum_val: Callable[[int], float]
    get_optimum_x: Callable[[int], np.ndarray]
    eval_vectorized: Callable[[np.ndarray], np.ndarray]

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evalúa la función sobre un individuo o una población.

        Args:
            x: Array de forma (n_vars,) para un individuo o (N, n_vars) para N individuos.

        Returns:
            Array de forma (N,) con los valores de aptitud bruta, o escalar float si la entrada es 1D.
        """
        arr = np.asarray(x, dtype=np.float64)
        is_single = arr.ndim == 1
        if is_single:
            arr_2d = arr.reshape(1, -1)
        else:
            arr_2d = arr

        res = self.eval_vectorized(arr_2d)

        if is_single:
            return float(res[0])
        return res

    def get_bounds_array(self, n_vars: int) -> np.ndarray:
        """Devuelve array de límites [[inf_0, sup_0], ...] de forma (n_vars, 2)."""
        return np.tile(np.array(self.bounds, dtype=np.float64), (n_vars, 1))


# ---------------------------------------------------------------------------
# 1. MAXIMIZACIÓN (1 variable)
# ---------------------------------------------------------------------------


def _eval_m1(x: np.ndarray) -> np.ndarray:
    # M1: f(x) = sen^6(5*pi*x) * exp(-2*ln(2)*((x-0.1)/0.8)^2)
    xi = x[:, 0]
    sin_term = np.sin(5.0 * np.pi * xi) ** 6
    exp_term = np.exp(-2.0 * np.log(2.0) * (((xi - 0.1) / 0.8) ** 2))
    return sin_term * exp_term


F_M1 = TargetFunction(
    id="M1",
    name="M1: Seno amortiguado",
    category="max_1d",
    target_type="max",
    formula_latex=r"f(x) = \sin^6(5\pi x) \cdot \exp\left(-2\ln(2)\left(\frac{x - 0.1}{0.8}\right)^2\right)",
    formula_display="f(x) = sen⁶(5πx) · exp(−2·ln(2)·((x − 0.1)/0.8)²)",
    default_n_vars=1,
    is_n_configurable=False,
    bounds=(0.0, 1.0),
    get_optimum_val=lambda n=1: 1.0,
    get_optimum_x=lambda n=1: np.array([0.1], dtype=np.float64),
    eval_vectorized=_eval_m1,
)


def _eval_m2(x: np.ndarray) -> np.ndarray:
    # M2: f(x) = 20 + e - 20*exp(-0.2*|x|) - exp(cos(2*pi*x))
    xi = x[:, 0]
    return 20.0 + np.e - 20.0 * np.exp(-0.2 * np.abs(xi)) - np.exp(np.cos(2.0 * np.pi * xi))


F_M2 = TargetFunction(
    id="M2",
    name="M2: Tipo Ackley 1D",
    category="max_1d",
    target_type="max",
    formula_latex=r"f(x) = 20 + e - 20\exp(-0.2|x|) - \exp(\cos(2\pi x))",
    formula_display="f(x) = 20 + e − 20·exp(−0.2·|x|) − exp(cos(2πx))",
    default_n_vars=1,
    is_n_configurable=False,
    bounds=(0.0, 32.0),
    get_optimum_val=lambda n=1: 22.3136,
    get_optimum_x=lambda n=1: np.array([31.5005], dtype=np.float64),
    eval_vectorized=_eval_m2,
)


def _eval_m3(x: np.ndarray) -> np.ndarray:
    # M3: f(x) = x + |sen(32*pi*x)|
    xi = x[:, 0]
    return xi + np.abs(np.sin(32.0 * np.pi * xi))


F_M3 = TargetFunction(
    id="M3",
    name="M3: Multimodal oscilante",
    category="max_1d",
    target_type="max",
    formula_latex=r"f(x) = x + |\sin(32\pi x)|",
    formula_display="f(x) = x + |sen(32πx)|",
    default_n_vars=1,
    is_n_configurable=False,
    bounds=(0.0, 1.0),
    get_optimum_val=lambda n=1: 1.98442447,
    get_optimum_x=lambda n=1: np.array([0.98447395], dtype=np.float64),
    eval_vectorized=_eval_m3,
)


def _eval_m4(x: np.ndarray) -> np.ndarray:
    # M4: f(x) = 0.5 - (sen^2(x) - 0.5) / (1 + 0.001*x^4)
    xi = x[:, 0]
    return 0.5 - (np.sin(xi) ** 2 - 0.5) / (1.0 + 0.001 * (xi**4))


F_M4 = TargetFunction(
    id="M4",
    name="M4: Racional trigonométrica",
    category="max_1d",
    target_type="max",
    formula_latex=r"f(x) = 0.5 - \frac{\sin^2(x) - 0.5}{1 + 0.001 x^4}",
    formula_display="f(x) = 0.5 − (sen²(x) − 0.5) / (1 + 0.001·x⁴)",
    default_n_vars=1,
    is_n_configurable=False,
    bounds=(0.0, 10.0),
    get_optimum_val=lambda n=1: 1.0,
    get_optimum_x=lambda n=1: np.array([0.0], dtype=np.float64),
    eval_vectorized=_eval_m4,
)


# ---------------------------------------------------------------------------
# 2. MINIMIZACIÓN (1 variable)
# ---------------------------------------------------------------------------


def _eval_min1(x: np.ndarray) -> np.ndarray:
    # m1: f(x) = 1 + cos(x) / (1 + 0.01*x^2)
    xi = x[:, 0]
    return 1.0 + np.cos(xi) / (1.0 + 0.01 * (xi**2))


F_MIN1 = TargetFunction(
    id="m1",
    name="m1: Coseno amortiguado",
    category="min_1d",
    target_type="min",
    formula_latex=r"f(x) = 1 + \frac{\cos(x)}{1 + 0.01 x^2}",
    formula_display="f(x) = 1 + cos(x) / (1 + 0.01·x²)",
    default_n_vars=1,
    is_n_configurable=False,
    bounds=(0.0, 50.0),
    get_optimum_val=lambda n=1: 0.0883634,
    get_optimum_x=lambda n=1: np.array([3.08531], dtype=np.float64),
    eval_vectorized=_eval_min1,
)


def _eval_min2(x: np.ndarray) -> np.ndarray:
    # m2: f(x) = -|x * sen(sqrt(|x|))|
    xi = x[:, 0]
    return -np.abs(xi * np.sin(np.sqrt(np.abs(xi))))


F_MIN2 = TargetFunction(
    id="m2",
    name="m2: Schwefel 1D simétrica",
    category="min_1d",
    target_type="min",
    formula_latex=r"f(x) = -|x \cdot \sin(\sqrt{|x|})|",
    formula_display="f(x) = −|x · sen(√|x|)|",
    default_n_vars=1,
    is_n_configurable=False,
    bounds=(-250.0, 250.0),
    get_optimum_val=lambda n=1: -201.843,
    get_optimum_x=lambda n=1: np.array([203.814], dtype=np.float64),
    eval_vectorized=_eval_min2,
)


def _eval_min3(x: np.ndarray) -> np.ndarray:
    # m3: f(x) = sen(x) / (1 + sqrt(x) + cos(x)/(1+x))
    xi = x[:, 0]
    den = 1.0 + np.sqrt(np.maximum(xi, 0.0)) + np.cos(xi) / (1.0 + xi)
    return np.sin(xi) / den


F_MIN3 = TargetFunction(
    id="m3",
    name="m3: Seno modulado",
    category="min_1d",
    target_type="min",
    formula_latex=r"f(x) = \frac{\sin(x)}{1 + \sqrt{x} + \frac{\cos(x)}{1 + x}}",
    formula_display="f(x) = sen(x) / (1 + √x + cos(x)/(1 + x))",
    default_n_vars=1,
    is_n_configurable=False,
    bounds=(0.0, 25.0),
    get_optimum_val=lambda n=1: -0.318071,
    get_optimum_x=lambda n=1: np.array([4.5798], dtype=np.float64),
    eval_vectorized=_eval_min3,
)


def _eval_min4(x: np.ndarray) -> np.ndarray:
    # m4 corregida del libro: 5 + exp(0.5*sen(pi*x^2)) - exp(0.1*x*cos(pi*x))
    xi = x[:, 0]
    return 5.0 + np.exp(0.5 * np.sin(np.pi * (xi**2))) - np.exp(0.1 * xi * np.cos(np.pi * xi))


F_MIN4 = TargetFunction(
    id="m4",
    name="m4: Doble exponencial corregida",
    category="min_1d",
    target_type="min",
    formula_latex=r"f(x) = 5 + \exp(0.5\sin(\pi x^2)) - \exp(0.1 x \cos(\pi x))",
    formula_display="f(x) = 5 + exp(0.5·sen(πx²)) − exp(0.1·x·cos(πx))",
    default_n_vars=1,
    is_n_configurable=False,
    bounds=(0.0, 25.0),
    get_optimum_val=lambda n=1: -5.39204,
    get_optimum_x=lambda n=1: np.array([23.9901], dtype=np.float64),
    eval_vectorized=_eval_min4,
)


# ---------------------------------------------------------------------------
# 3. VARIAS VARIABLES
# ---------------------------------------------------------------------------


def _eval_v1(x: np.ndarray) -> np.ndarray:
    # V1 (Schaffer F6 en 2D):
    # f(x1, x2) = 0.5 - (-0.5 + sen(sqrt(x1^2 + x2^2))^2) / (1 + 0.001*(x1^2+x2^2))^2
    r2 = x[:, 0] ** 2 + x[:, 1] ** 2
    r = np.sqrt(r2)
    num = (np.sin(r) ** 2) - 0.5
    den = (1.0 + 0.001 * r2) ** 2
    return 0.5 - (num / den)


F_V1 = TargetFunction(
    id="V1",
    name="V1: Schaffer 2D",
    category="multi",
    target_type="max",
    formula_latex=r"f(x_1,x_2) = 0.5 - \frac{\sin^2(\sqrt{x_1^2+x_2^2}) - 0.5}{(1 + 0.001(x_1^2+x_2^2))^2}",
    formula_display="f(x₁,x₂) = 0.5 − (−0.5 + sen²(√(x₁²+x₂²))) / (1 + 0.001·(x₁²+x₂²))²",
    default_n_vars=2,
    is_n_configurable=False,
    bounds=(-40.0, 40.0),
    get_optimum_val=lambda n=2: 1.0,
    get_optimum_x=lambda n=2: np.zeros(2, dtype=np.float64),
    eval_vectorized=_eval_v1,
)


def _eval_v2(x: np.ndarray) -> np.ndarray:
    # V2 (Rastrigin): f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i))
    n = x.shape[1]
    sum_terms = np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=1)
    return 10.0 * n + sum_terms


F_V2 = TargetFunction(
    id="V2",
    name="V2: Rastrigin",
    category="multi",
    target_type="min",
    formula_latex=r"f(x) = 10n + \sum_{i=1}^n \left(x_i^2 - 10\cos(2\pi x_i)\right)",
    formula_display="f(x) = 10n + Σᵢ (xᵢ² − 10·cos(2πxᵢ))",
    default_n_vars=5,
    is_n_configurable=True,
    bounds=(-5.12, 5.12),
    get_optimum_val=lambda n: 0.0,
    get_optimum_x=lambda n: np.zeros(n, dtype=np.float64),
    eval_vectorized=_eval_v2,
)


# Mínimos teóricos comprobados para Michalewicz (V3) según n (índice 1..n con factor (i+1))
V3_MINIMA_TABLE: dict[int, float] = {
    1: -1.0,
    2: -1.959091,
    3: -2.897553,
    4: -3.886358,
    5: -4.886358,
    6: -5.879585,
    7: -6.862457,
    8: -7.858851,
    9: -8.858851,
    10: -9.856182,
}


def _get_v3_optimum_val(n: int) -> float:
    if n in V3_MINIMA_TABLE:
        return V3_MINIMA_TABLE[n]
    # Aproximación lineal estimada para n > 10 basada en la pendiente promedio (~ -0.985 por dimensión)
    return -0.985 * n


def _eval_v3(x: np.ndarray) -> np.ndarray:
    # V3 (Michalewicz del libro): f(x) = -sum_{i=1}^n sen(x_i) * sen^20( (i+1)*x_i^2 / pi )
    # En Python, el índice k va de 0 a n-1, así que i+1 = (k+1)+1 = k+2
    n = x.shape[1]
    k_indices = np.arange(1, n + 1, dtype=np.float64)  # 1, 2, ..., n
    factors = (k_indices + 1.0) / np.pi  # (i+1)/pi
    # x shape: (N, n)
    sin_x = np.sin(x)
    sin_inner = np.sin(factors * (x**2))
    terms = sin_x * (sin_inner**20)
    return -np.sum(terms, axis=1)


F_V3 = TargetFunction(
    id="V3",
    name="V3: Michalewicz",
    category="multi",
    target_type="min",
    formula_latex=r"f(x) = -\sum_{i=1}^n \sin(x_i) \cdot \sin^{20}\left( \frac{(i+1) x_i^2}{\pi} \right)",
    formula_display="f(x) = −Σᵢ₌₁ⁿ sen(xᵢ) · sen²⁰( (i+1)·xᵢ² / π )",
    default_n_vars=5,
    is_n_configurable=True,
    bounds=(0.0, np.pi),
    get_optimum_val=_get_v3_optimum_val,
    get_optimum_x=lambda n: np.full(n, 1.5707963, dtype=np.float64),
    eval_vectorized=_eval_v3,
)


def _eval_v4(x: np.ndarray) -> np.ndarray:
    # V4: f(x) = sum_{i=1}^n -x_i * sen(sqrt(|x_i|))
    terms = -x * np.sin(np.sqrt(np.abs(x)))
    return np.sum(terms, axis=1)


F_V4 = TargetFunction(
    id="V4",
    name="V4: Schwefel multivariable",
    category="multi",
    target_type="min",
    formula_latex=r"f(x) = \sum_{i=1}^n -x_i \cdot \sin(\sqrt{|x_i|})",
    formula_display="f(x) = Σᵢ −xᵢ·sen(√|xᵢ|)",
    default_n_vars=5,
    is_n_configurable=True,
    bounds=(0.0, 100.0),
    get_optimum_val=lambda n: -float(n) * 63.63498,
    get_optimum_x=lambda n: np.full(n, 65.54785, dtype=np.float64),
    eval_vectorized=_eval_v4,
)


# Catálogo completo indexado por ID
CATALOG: dict[str, TargetFunction] = {
    "M1": F_M1,
    "M2": F_M2,
    "M3": F_M3,
    "M4": F_M4,
    "m1": F_MIN1,
    "m2": F_MIN2,
    "m3": F_MIN3,
    "m4": F_MIN4,
    "V1": F_V1,
    "V2": F_V2,
    "V3": F_V3,
    "V4": F_V4,
}

ALL_FUNCTIONS: list[TargetFunction] = list(CATALOG.values())


def get_function(func_id: str) -> TargetFunction:
    """Obtiene una función del catálogo por su ID ('M1', 'm2', 'V2', etc.)."""
    if func_id not in CATALOG:
        raise KeyError(f"Función '{func_id}' no encontrada en el catálogo. Disponibles: {list(CATALOG.keys())}")
    return CATALOG[func_id]
