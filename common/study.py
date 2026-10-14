"""Módulo de utilidades para estudios paramétricos y comparativas estadísticas.

Permite ejecutar experimentos sistemáticos fijando parámetros base y variando uno a la vez,
recopilando métricas (media, desviación típica, mediana, error, generaciones hasta el óptimo,
tasa de éxito y tiempo de CPU), exportando a CSV y generando gráficas con Matplotlib.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para scripts y exportación
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.functions import TargetFunction, get_function


@dataclass
class RunMetrics:
    """Métricas individuales de una sola ejecución."""

    seed: int
    final_best_raw: float
    error_to_optimum: float
    reached_optimum: bool
    generation_reached: int | None
    cpu_time_seconds: float
    convergence_best: list[float]
    convergence_mean: list[float]


@dataclass
class ParamExperimentSummary:
    """Resumen estadístico de N ejecuciones para un valor de parámetro concreto."""

    param_name: str
    param_value: Any
    func_id: str
    n_runs: int
    # Estadísticas de mejor valor final
    best_mean: float
    best_std: float
    best_median: float
    best_min: float
    best_max: float
    # Estadísticas de error
    error_mean: float
    error_std: float
    error_median: float
    error_min: float
    error_max: float
    # Éxito y convergencia
    success_rate_pct: float
    avg_gen_reached: float | None
    # Tiempo
    avg_cpu_time: float
    # Curva promedio de convergencia
    mean_convergence: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "param_name": self.param_name,
            "param_value": str(self.param_value),
            "func_id": self.func_id,
            "n_runs": self.n_runs,
            "best_mean": self.best_mean,
            "best_std": self.best_std,
            "best_median": self.best_median,
            "best_min": self.best_min,
            "best_max": self.best_max,
            "error_mean": self.error_mean,
            "error_std": self.error_std,
            "error_median": self.error_median,
            "error_min": self.error_min,
            "error_max": self.error_max,
            "success_rate_pct": self.success_rate_pct,
            "avg_gen_reached": self.avg_gen_reached if self.avg_gen_reached is not None else -1,
            "avg_cpu_time": self.avg_cpu_time,
        }


def compute_experiment_summary(
    param_name: str,
    param_value: Any,
    func_id: str,
    runs: list[RunMetrics],
) -> ParamExperimentSummary:
    """Calcula las métricas agregadas a partir de una lista de ejecuciones."""
    n_runs = len(runs)
    bests = np.array([r.final_best_raw for r in runs], dtype=np.float64)
    errors = np.array([r.error_to_optimum for r in runs], dtype=np.float64)
    cpu_times = np.array([r.cpu_time_seconds for r in runs], dtype=np.float64)
    successes = [r.reached_optimum for r in runs]
    gens_reached = [r.generation_reached for r in runs if r.generation_reached is not None]

    # Calcular curva de convergencia media
    max_len = max(len(r.convergence_best) for r in runs)
    conv_matrix = np.full((n_runs, max_len), np.nan, dtype=np.float64)
    for i, r in enumerate(runs):
        arr = np.array(r.convergence_best)
        conv_matrix[i, : len(arr)] = arr
        if len(arr) < max_len:
            conv_matrix[i, len(arr) :] = arr[-1]

    mean_conv = np.nanmean(conv_matrix, axis=0)

    return ParamExperimentSummary(
        param_name=param_name,
        param_value=param_value,
        func_id=func_id,
        n_runs=n_runs,
        best_mean=float(np.mean(bests)),
        best_std=float(np.std(bests, ddof=1)) if n_runs > 1 else 0.0,
        best_median=float(np.median(bests)),
        best_min=float(np.min(bests)),
        best_max=float(np.max(bests)),
        error_mean=float(np.mean(errors)),
        error_std=float(np.std(errors, ddof=1)) if n_runs > 1 else 0.0,
        error_median=float(np.median(errors)),
        error_min=float(np.min(errors)),
        error_max=float(np.max(errors)),
        success_rate_pct=float(np.mean(successes) * 100.0),
        avg_gen_reached=float(np.mean(gens_reached)) if gens_reached else None,
        avg_cpu_time=float(np.mean(cpu_times)),
        mean_convergence=mean_conv,
    )


def save_summaries_to_csv(summaries: Sequence[ParamExperimentSummary], filepath: str | Path) -> None:
    """Guarda una lista de resúmenes experimentales a un CSV."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([s.to_dict() for s in summaries])
    df.to_csv(path, index=False)


def plot_param_variation_results(
    summaries: Sequence[ParamExperimentSummary],
    param_name: str,
    func_name: str,
    output_path: str | Path,
) -> None:
    """Genera una figura comparativa de alta calidad mostrando error y convergencia por valor del parámetro."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    plt.subplots_adjust(wspace=0.3)

    param_labels = [str(s.param_value) for s in summaries]
    means = [s.error_mean for s in summaries]
    stds = [s.error_std for s in summaries]
    x_pos = np.arange(len(param_labels))

    # Gráfico 1: Barras con barras de error del error respecto al óptimo
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, color="#3b82f6", alpha=0.85, edgecolor="#1d4ed8", linewidth=1.2)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(param_labels)
    ax1.set_xlabel(f"Valor de {param_name}", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Error absoluto (|f - f*|)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Impacto de {param_name} en {func_name}\n(Error: media ± desv. típ.)", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Etiquetas sobre barras con tasa de éxito
    for i, s in enumerate(summaries):
        ax1.annotate(
            f"Éxito: {s.success_rate_pct:.0f}%",
            xy=(x_pos[i], means[i] + stds[i]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="semibold",
            color="#0f172a",
        )

    # Gráfico 2: Curvas de convergencia media por valor
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(summaries)))
    for s, c in zip(summaries, colors):
        curve = s.mean_convergence
        gens = np.arange(len(curve))
        ax2.plot(gens, curve, label=f"{param_name}={s.param_value}", color=c, linewidth=2.0)

    ax2.set_xlabel("Generación", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Aptitud bruta media", fontsize=11, fontweight="bold")
    ax2.set_title("Curvas medias de convergencia", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="best", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_elitism_comparison(
    summaries_elite: Sequence[ParamExperimentSummary],
    summaries_no_elite: Sequence[ParamExperimentSummary],
    func_ids: Sequence[str],
    output_path: str | Path,
) -> None:
    """Genera una gráfica comparativa directa Con Elitismo vs Sin Elitismo."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)

    x = np.arange(len(func_ids))
    width = 0.35

    elite_err = [s.error_mean for s in summaries_elite]
    elite_err_std = [s.error_std for s in summaries_elite]
    no_elite_err = [s.error_mean for s in summaries_no_elite]
    no_elite_err_std = [s.error_std for s in summaries_no_elite]

    # Panel 1: Error medio
    ax1.bar(x - width / 2, elite_err, width, yerr=elite_err_std, capsize=4, label="Con Elitismo (2%)", color="#10b981", alpha=0.9)
    ax1.bar(x + width / 2, no_elite_err, width, yerr=no_elite_err_std, capsize=4, label="Sin Elitismo", color="#ef4444", alpha=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(func_ids, fontweight="bold")
    ax1.set_ylabel("Error absoluto (|f - f*|)", fontsize=11, fontweight="bold")
    ax1.set_title("Calidad final: Con Elitismo vs Sin Elitismo", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Panel 2: Tasa de éxito
    elite_succ = [s.success_rate_pct for s in summaries_elite]
    no_elite_succ = [s.success_rate_pct for s in summaries_no_elite]
    ax2.bar(x - width / 2, elite_succ, width, label="Con Elitismo (2%)", color="#10b981", alpha=0.9)
    ax2.bar(x + width / 2, no_elite_succ, width, label="Sin Elitismo", color="#ef4444", alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(func_ids, fontweight="bold")
    ax2.set_ylabel("% Éxito (alcance de óptimo)", fontsize=11, fontweight="bold")
    ax2.set_title("Tasa de éxito (%): Con Elitismo vs Sin Elitismo", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
