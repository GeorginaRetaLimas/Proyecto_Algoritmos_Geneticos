"""Estructuras de datos y utilidades para estadísticas por generación y exportación."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class GenerationStats:
    """Estadísticas de una generación producidas por el generador del AG."""

    generation: int
    best_fitness_raw: float  # mejor valor REAL de la función en la generación
    mean_fitness_raw: float
    worst_fitness_raw: float
    best_so_far_raw: float  # mejor valor histórico acumulado
    best_x: np.ndarray  # fenotipo del mejor de la generación (n_vars,)
    population_x: np.ndarray  # fenotipos de toda la población (tam_pob, n_vars)
    population_fitness_raw: np.ndarray  # aptitudes brutas (tam_pob,)
    best_so_far_x: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def to_row(self) -> dict[str, float | int | str]:
        """Convierte las estadísticas escalares a un diccionario para exportar a CSV/DataFrame."""
        row: dict[str, float | int | str] = {
            "generation": self.generation,
            "best_fitness_raw": float(self.best_fitness_raw),
            "mean_fitness_raw": float(self.mean_fitness_raw),
            "worst_fitness_raw": float(self.worst_fitness_raw),
            "best_so_far_raw": float(self.best_so_far_raw),
        }
        # Agregar variables individuales de best_x
        for i, val in enumerate(self.best_x):
            row[f"best_x_{i+1}"] = float(val)
        return row


def export_history_to_csv(history: Sequence[GenerationStats], filepath: str | Path) -> None:
    """Exporta el historial completo de generaciones a un archivo CSV."""
    if not history:
        return

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [s.to_row() for s in history]
    fieldnames = list(rows[0].keys())

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
