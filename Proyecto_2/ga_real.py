"""Motor del Algoritmo Genético con representación real (Proyecto 2).

Optimiza funciones reales sin decodificación, permitiendo seleccionar entre 4 operadores
de cruce (un punto, discreto uniforme, aritmético, SBX), mutación uniforme y elitismo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from common.functions import TargetFunction, get_function
from common.stats import GenerationStats
from Proyecto_2.operadores import (
    CROSSOVER_OPERATORS,
    CrossoverType,
    mutate_uniform_population,
)


@dataclass
class RealGAConfig:
    """Configuración para el Algoritmo Genético de representación real."""

    function: TargetFunction
    n_vars: int = 1
    bounds: tuple[float, float] | Sequence[tuple[float, float]] | None = None
    tam_pob: int = 100
    num_gen: int = 500
    prob_cruce: float = 0.8
    prob_mut: float = 0.05  # por gen
    crossover_type: CrossoverType = "sbx"
    p_u: float = 0.5  # Cruce uniforme
    alpha: float | None = 0.5  # Cruce aritmético
    random_alpha_per_gene: bool = False
    n_sbx: float = 1.0  # Cruce SBX
    alpha_per_gene: bool = False
    elitismo: bool = True
    pct_elite: float = 2.0  # % de élite (2%)
    cmax_factor: float = 1.05
    seed: int | None = None


class RealGeneticAlgorithm:
    """Motor del Algoritmo Genético con representación real."""

    def __init__(self, config: RealGAConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.func = config.function
        self.n_vars = config.n_vars

        # Generador de números aleatorios reproducible
        if rng is not None:
            self.rng = rng
            self.used_seed = config.seed
        elif config.seed is not None:
            self.used_seed = int(config.seed)
            self.rng = np.random.default_rng(self.used_seed)
        else:
            self.used_seed = int(np.random.default_rng().integers(0, 2**31 - 1))
            self.rng = np.random.default_rng(self.used_seed)

        # Configurar límites por variable: shape (n_vars, 2)
        if config.bounds is None:
            self.bounds = self.func.get_bounds_array(self.n_vars)
        elif isinstance(config.bounds, tuple) and len(config.bounds) == 2 and isinstance(config.bounds[0], (int, float)):
            self.bounds = np.tile(np.array(config.bounds, dtype=np.float64), (self.n_vars, 1))
        else:
            self.bounds = np.array(config.bounds, dtype=np.float64)
            if self.bounds.shape != (self.n_vars, 2):
                raise ValueError(f"Los límites deben tener forma ({self.n_vars}, 2), obtenido: {self.bounds.shape}")

        # Cálculo de élite
        if self.config.elitismo:
            self.tam_elite = max(1, int(round(self.config.tam_pob * (self.config.pct_elite / 100.0))))
            self.tam_elite = min(self.tam_elite, self.config.tam_pob)
        else:
            self.tam_elite = 0

        # Obtener función de cruce seleccionada
        c_type = self.config.crossover_type
        if c_type not in CROSSOVER_OPERATORS:
            raise KeyError(f"Operador de cruce desconocido: {c_type}. Opciones: {list(CROSSOVER_OPERATORS.keys())}")
        self.crossover_func = CROSSOVER_OPERATORS[c_type]

    def _transform_fitness_for_minimization(self, raw_fitness: np.ndarray) -> np.ndarray:
        """Transformación de aptitud para minimización garantizando aptitudes positivas."""
        cmax = float(np.max(raw_fitness))
        if cmax > 0:
            cmax_ajustado = cmax * self.config.cmax_factor
        else:
            cmax_ajustado = cmax + (self.config.cmax_factor - 1.0) * abs(cmax) + 1e-6

        spread = cmax_ajustado - cmax
        if spread <= 1e-12:
            cmax_ajustado += 1e-4

        aptitud = cmax_ajustado - raw_fitness
        return np.maximum(aptitud, 1e-8)

    def _prepare_selection_fitness(self, raw_fitness: np.ndarray) -> np.ndarray:
        """Prepara aptitudes positivas proporcionales a la bondad para ruleta."""
        if self.func.target_type == "min":
            return self._transform_fitness_for_minimization(raw_fitness)

        fmin = float(np.min(raw_fitness))
        if fmin < 0:
            shift = abs(fmin) * 1.05 + 1e-6
            return raw_fitness + shift
        if fmin == 0:
            return raw_fitness + 1e-6

        return np.maximum(raw_fitness, 1e-8)

    def _select_roulette(self, population: np.ndarray, selection_fitness: np.ndarray) -> np.ndarray:
        """Selecciona individuos mediante ruleta, preservando la élite en las primeras posiciones."""
        tam_pob = self.config.tam_pob
        new_pop = np.empty_like(population)

        # 1. Élite
        if self.tam_elite > 0:
            elite_indices = np.argsort(selection_fitness)[-self.tam_elite :]
            new_pop[: self.tam_elite] = population[elite_indices]

        # 2. Ruleta para el resto
        n_needed = tam_pob - self.tam_elite
        if n_needed > 0:
            sum_fit = float(np.sum(selection_fitness))
            if sum_fit <= 1e-12 or not np.isfinite(sum_fit):
                probs = np.full(tam_pob, 1.0 / tam_pob, dtype=np.float64)
            else:
                probs = selection_fitness / sum_fit

            punt_acu = np.cumsum(probs)
            punt_acu[-1] = 1.0

            r = self.rng.random(n_needed)
            selected_indices = np.searchsorted(punt_acu, r)
            selected_indices = np.clip(selected_indices, 0, tam_pob - 1)
            new_pop[self.tam_elite :] = population[selected_indices]

        return new_pop

    def _crossover(self, population: np.ndarray) -> np.ndarray:
        """Aplica el operador de cruce seleccionado sobre los individuos no élite."""
        tam_pob = self.config.tam_pob
        non_elite_start = self.tam_elite
        n_non_elite = tam_pob - non_elite_start

        if n_non_elite < 2 or self.config.prob_cruce <= 0:
            return population

        roll = self.rng.random(n_non_elite)
        chosen_rel = np.where(roll < self.config.prob_cruce)[0]

        if len(chosen_rel) % 2 != 0:
            chosen_rel = chosen_rel[:-1]

        if len(chosen_rel) == 0:
            return population

        chosen_indices = non_elite_start + chosen_rel

        # Parámetros adicionales para el operador de cruce
        kwargs = {
            "p_u": self.config.p_u,
            "alpha": self.config.alpha,
            "random_alpha_per_gene": self.config.random_alpha_per_gene,
            "n_sbx": self.config.n_sbx,
            "alpha_per_gene": self.config.alpha_per_gene,
        }

        for idx in range(0, len(chosen_indices), 2):
            i1 = chosen_indices[idx]
            i2 = chosen_indices[idx + 1]

            p1 = population[i1]
            p2 = population[i2]

            h1, h2 = self.crossover_func(p1, p2, self.bounds, self.rng, **kwargs)
            population[i1] = h1
            population[i2] = h2

        return population

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        """Aplica mutación uniforme a los individuos no élite."""
        return mutate_uniform_population(
            population=population,
            bounds=self.bounds,
            prob_mut=self.config.prob_mut,
            non_elite_start=self.tam_elite,
            rng=self.rng,
        )

    def run(self) -> Iterator[GenerationStats]:
        """Ejecuta el Algoritmo Genético real como un generador de estadísticas."""
        tam_pob = self.config.tam_pob
        num_gen = self.config.num_gen
        is_max = self.func.target_type == "max"

        # 1. Población inicial: números reales uniformes en los límites de cada gen
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        pop_x = self.rng.uniform(lows, highs, size=(tam_pob, self.n_vars))

        # Evaluación directa sin decodificación
        pop_fitness_raw = self.func.evaluate(pop_x)

        if is_max:
            best_idx = int(np.argmax(pop_fitness_raw))
        else:
            best_idx = int(np.argmin(pop_fitness_raw))

        best_fit = float(pop_fitness_raw[best_idx])
        best_x = pop_x[best_idx].copy()
        worst_fit = float(np.min(pop_fitness_raw) if is_max else np.max(pop_fitness_raw))
        mean_fit = float(np.mean(pop_fitness_raw))

        best_so_far_raw = best_fit
        best_so_far_x = best_x.copy()

        # Emitir generación 0
        yield GenerationStats(
            generation=0,
            best_fitness_raw=best_fit,
            mean_fitness_raw=mean_fit,
            worst_fitness_raw=worst_fit,
            best_so_far_raw=best_so_far_raw,
            best_x=best_x,
            population_x=pop_x,
            population_fitness_raw=pop_fitness_raw,
            best_so_far_x=best_so_far_x,
        )

        # 2. Bucle generacional (1 .. num_gen)
        for gen in range(1, num_gen + 1):
            selection_fit = self._prepare_selection_fitness(pop_fitness_raw)

            # Selección por ruleta con élite
            pop_x = self._select_roulette(pop_x, selection_fit)

            # Cruce
            pop_x = self._crossover(pop_x)

            # Mutación uniforme
            pop_x = self._mutate(pop_x)

            # Reevaluación directa
            pop_fitness_raw = self.func.evaluate(pop_x)

            if is_max:
                best_idx = int(np.argmax(pop_fitness_raw))
            else:
                best_idx = int(np.argmin(pop_fitness_raw))

            best_fit = float(pop_fitness_raw[best_idx])
            best_x = pop_x[best_idx].copy()
            worst_fit = float(np.min(pop_fitness_raw) if is_max else np.max(pop_fitness_raw))
            mean_fit = float(np.mean(pop_fitness_raw))

            # Actualizar mejor histórico
            if is_max:
                if best_fit > best_so_far_raw:
                    best_so_far_raw = best_fit
                    best_so_far_x = best_x.copy()
            else:
                if best_fit < best_so_far_raw:
                    best_so_far_raw = best_fit
                    best_so_far_x = best_x.copy()

            yield GenerationStats(
                generation=gen,
                best_fitness_raw=best_fit,
                mean_fitness_raw=mean_fit,
                worst_fitness_raw=worst_fit,
                best_so_far_raw=best_so_far_raw,
                best_x=best_x,
                population_x=pop_x,
                population_fitness_raw=pop_fitness_raw,
                best_so_far_x=best_so_far_x,
            )
