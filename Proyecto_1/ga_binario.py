"""Motor del Algoritmo Genético con representación binaria (Proyecto 1).

Implementa la optimización de funciones monovariables y multivariables mediante
cadenas binarias por variable, selección por ruleta, cruce monopunto, mutación bit a bit
y elitismo configurable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import numpy as np

from common.functions import TargetFunction, get_function
from common.stats import GenerationStats


@dataclass
class BinaryGAConfig:
    """Configuración para el Algoritmo Genético binario."""

    function: TargetFunction
    n_vars: int = 1
    bounds: tuple[float, float] | Sequence[tuple[float, float]] | None = None
    tol: float = 1e-4
    tam_pob: int = 500
    num_gen: int = 1000
    prob_cruce: float = 0.4
    prob_mut: float = 0.001
    elitismo: bool = True
    pct_elite: float = 2.0  # porcentaje (2.0 = 2 %)
    cmax_factor: float = 1.05  # factor para minimización (1.05 según libro)
    seed: int | None = None


class BinaryGeneticAlgorithm:
    """Motor del Algoritmo Genético con representación binaria.

    El algoritmo opera como un generador (método `run()`), emitiendo un objeto
    `GenerationStats` tras la generación 0 (población inicial) y tras cada generación
    evolucionada.
    """

    def __init__(self, config: BinaryGAConfig, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.func = config.function
        self.n_vars = config.n_vars

        # Inicialización del generador aleatorio reproducible
        if rng is not None:
            self.rng = rng
            self.used_seed = config.seed
        elif config.seed is not None:
            self.used_seed = int(config.seed)
            self.rng = np.random.default_rng(self.used_seed)
        else:
            # Generar semilla aleatoria visible
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

        # Cálculo de longitud de cromosoma por variable (lcrom) y offsets
        # lcrom = ceil( log2( 1 + (x_max - x_min) / TOL ) )
        self.lcrom: list[int] = []
        for j in range(self.n_vars):
            x_min, x_max = self.bounds[j]
            ratio = 1.0 + (x_max - x_min) / max(self.config.tol, 1e-12)
            length = int(math.ceil(math.log2(max(ratio, 1.0))))
            length = max(length, 1)  # Al menos 1 bit
            self.lcrom.append(length)

        self.total_bits = sum(self.lcrom)

        # Offsets para rebanar el cromosoma contiguo por variable
        self.var_slices: list[slice] = []
        # Multiplicadores precalculados para decodificación binaria: 2^i con i=0..lcrom-1
        self.powers_of_two: list[np.ndarray] = []
        self.max_dec_values: list[float] = []

        start = 0
        for length in self.lcrom:
            end = start + length
            self.var_slices.append(slice(start, end))
            powers = 2.0 ** np.arange(length, dtype=np.float64)
            self.powers_of_two.append(powers)
            self.max_dec_values.append(float(2**length - 1))
            start = end

        # Cálculo de tamaño de élite
        if self.config.elitismo:
            self.tam_elite = max(1, int(round(self.config.tam_pob * (self.config.pct_elite / 100.0))))
            # La élite no puede ser mayor que la población
            self.tam_elite = min(self.tam_elite, self.config.tam_pob)
        else:
            self.tam_elite = 0

    def decode_population(self, population_chroms: np.ndarray) -> np.ndarray:
        """Decodifica una matriz de cromosomas binarios a fenotipos reales.

        Args:
            population_chroms: Array bool o uint8 de forma (tam_pob, total_bits).

        Returns:
            Array float64 de forma (tam_pob, n_vars).
        """
        tam_pob = population_chroms.shape[0]
        pop_x = np.empty((tam_pob, self.n_vars), dtype=np.float64)

        for j in range(self.n_vars):
            s = self.var_slices[j]
            bits = population_chroms[:, s].astype(np.float64)
            powers = self.powers_of_two[j]
            # bin_dec = sum(bits[i] * 2^i) donde el gen 0 es el bit menos significativo
            bin_dec = np.dot(bits, powers)
            x_min, x_max = self.bounds[j]
            max_dec = self.max_dec_values[j]
            if max_dec > 0:
                pop_x[:, j] = x_min + (x_max - x_min) * (bin_dec / max_dec)
            else:
                pop_x[:, j] = x_min

        return pop_x

    def _transform_fitness_for_minimization(self, raw_fitness: np.ndarray) -> np.ndarray:
        """Transforma aptitud bruta para minimización.

        Aptitud transformada f = cmax_ajustado - g(x).
        Corrige el caso cmax <= 0 y homogeneidad para garantizar aptitudes estrictamente positivas.
        """
        cmax = float(np.max(raw_fitness))
        cmin = float(np.min(raw_fitness))

        # Corrección de libro C++: si cmax <= 0, usar cmax + 0.05*|cmax| + delta
        if cmax > 0:
            cmax_ajustado = cmax * self.config.cmax_factor
        else:
            cmax_ajustado = cmax + (self.config.cmax_factor - 1.0) * abs(cmax) + 1e-6

        # Si toda la población tiene el mismo valor
        spread = cmax_ajustado - cmax
        if spread <= 1e-12:
            cmax_ajustado += 1e-4

        aptitud = cmax_ajustado - raw_fitness
        return np.maximum(aptitud, 1e-8)

    def _prepare_selection_fitness(self, raw_fitness: np.ndarray) -> np.ndarray:
        """Calcula las aptitudes adaptadas no negativas para la selección por ruleta."""
        if self.func.target_type == "min":
            return self._transform_fitness_for_minimization(raw_fitness)

        # Caso maximización: asegurar que todos los valores sean estrictamente positivos
        fmin = float(np.min(raw_fitness))
        if fmin < 0:
            # Desplazar para que el mínimo sea positivo
            shift = abs(fmin) * 1.05 + 1e-6
            return raw_fitness + shift

        # Si todos son positivos pero el mínimo es 0
        if fmin == 0:
            return raw_fitness + 1e-6

        return np.maximum(raw_fitness, 1e-8)

    def _select_roulette(self, population: np.ndarray, selection_fitness: np.ndarray) -> np.ndarray:
        """Selecciona individuos mediante ruleta, preservando la élite en las primeras posiciones."""
        tam_pob = self.config.tam_pob
        new_pop = np.empty_like(population)

        # 1. Identificar y colocar la élite
        if self.tam_elite > 0:
            # Índices de los mejores individuos según la aptitud de selección (o aptitud bruta)
            elite_indices = np.argsort(selection_fitness)[-self.tam_elite :]
            new_pop[: self.tam_elite] = population[elite_indices]

        # 2. Ruleta para el resto: tam_pob - tam_elite individuos
        n_needed = tam_pob - self.tam_elite
        if n_needed > 0:
            sum_fit = float(np.sum(selection_fitness))
            if sum_fit <= 1e-12 or not np.isfinite(sum_fit):
                probs = np.full(tam_pob, 1.0 / tam_pob, dtype=np.float64)
            else:
                probs = selection_fitness / sum_fit

            punt_acu = np.cumsum(probs)
            punt_acu[-1] = 1.0  # Proteger contra imprecisiones de redondeo

            r = self.rng.random(n_needed)
            selected_indices = np.searchsorted(punt_acu, r)
            # Clip de seguridad
            selected_indices = np.clip(selected_indices, 0, tam_pob - 1)
            new_pop[self.tam_elite :] = population[selected_indices]

        return new_pop

    def _crossover(self, population: np.ndarray) -> np.ndarray:
        """Aplica cruce monopunto por variable sobre los individuos no élite seleccionados."""
        tam_pob = self.config.tam_pob
        # La élite nunca es reemplazada por cruce
        non_elite_start = self.tam_elite
        n_non_elite = tam_pob - non_elite_start

        if n_non_elite < 2 or self.config.prob_cruce <= 0:
            return population

        # Decidir qué individuos no élite participan en cruce
        roll = self.rng.random(n_non_elite)
        chosen_relative = np.where(roll < self.config.prob_cruce)[0]

        # Si el número de seleccionados es impar, se descarta el último para hacerlo par
        if len(chosen_relative) % 2 != 0:
            chosen_relative = chosen_relative[:-1]

        if len(chosen_relative) == 0:
            return population

        chosen_indices = non_elite_start + chosen_relative

        # Emparejar consecutivamente: (chosen[0], chosen[1]), (chosen[2], chosen[3]), ...
        for idx in range(0, len(chosen_indices), 2):
            i1 = chosen_indices[idx]
            i2 = chosen_indices[idx + 1]

            padre1 = population[i1].copy()
            padre2 = population[i2].copy()

            hijo1 = padre1.copy()
            hijo2 = padre2.copy()

            # Para cada variable, se genera un punto de corte independiente (corrección al C++)
            for j in range(self.n_vars):
                s = self.var_slices[j]
                length = self.lcrom[j]
                if length > 1:
                    # Punto de corte en [1, length - 1] (corrección al C++)
                    p = int(self.rng.integers(1, length))
                    sub1 = padre1[s]
                    sub2 = padre2[s]
                    hijo1[s] = np.concatenate([sub1[:p], sub2[p:]])
                    hijo2[s] = np.concatenate([sub2[:p], sub1[p:]])

            population[i1] = hijo1
            population[i2] = hijo2

        return population

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        """Aplica mutación bit a bit a los individuos no élite."""
        if self.config.prob_mut <= 0:
            return population

        tam_pob = self.config.tam_pob
        non_elite_start = self.tam_elite
        n_non_elite = tam_pob - non_elite_start

        if n_non_elite <= 0:
            return population

        # Máscara booleana de mutación para los individuos no élite
        mutation_mask = self.rng.random((n_non_elite, self.total_bits)) < self.config.prob_mut
        population[non_elite_start:] ^= mutation_mask.astype(np.uint8)

        return population

    def run(self) -> Iterator[GenerationStats]:
        """Ejecuta el Algoritmo Genético como generador.

        Yields:
            GenerationStats tras la generación 0 (población inicial) y tras cada
            generación completada hasta num_gen.
        """
        tam_pob = self.config.tam_pob
        num_gen = self.config.num_gen
        is_max = self.func.target_type == "max"

        # 1. Generación 0: Población inicial aleatoria (bits 0 o 1 con prob 50/50)
        pop_chroms = self.rng.integers(0, 2, size=(tam_pob, self.total_bits), dtype=np.uint8)

        # Decodificación y evaluación inicial
        pop_x = self.decode_population(pop_chroms)
        pop_fitness_raw = self.func.evaluate(pop_x)

        # Mejor de la generación inicial
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

        # Emitir estadísticas generación 0
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

        # 2. Bucle principal de evolución (1 .. num_gen)
        for gen in range(1, num_gen + 1):
            # Preparar aptitudes de selección (siempre mayores cuanto mejor es el individuo)
            selection_fit = self._prepare_selection_fitness(pop_fitness_raw)

            # Selección por ruleta con élite
            pop_chroms = self._select_roulette(pop_chroms, selection_fit)

            # Reproducción / cruce monopunto
            pop_chroms = self._crossover(pop_chroms)

            # Mutación bit a bit
            pop_chroms = self._mutate(pop_chroms)

            # Reevaluación completa de la población
            pop_x = self.decode_population(pop_chroms)
            pop_fitness_raw = self.func.evaluate(pop_x)

            # Estadísticas de la generación
            if is_max:
                best_idx = int(np.argmax(pop_fitness_raw))
            else:
                best_idx = int(np.argmin(pop_fitness_raw))

            best_fit = float(pop_fitness_raw[best_idx])
            best_x = pop_x[best_idx].copy()
            worst_fit = float(np.min(pop_fitness_raw) if is_max else np.max(pop_fitness_raw))
            mean_fit = float(np.mean(pop_fitness_raw))

            # Actualizar mejor histórico acumulado
            if is_max:
                if best_fit > best_so_far_raw:
                    best_so_far_raw = best_fit
                    best_so_far_x = best_x.copy()
            else:
                if best_fit < best_so_far_raw:
                    best_so_far_raw = best_fit
                    best_so_far_x = best_x.copy()

            # Emitir estadísticas
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
