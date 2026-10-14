"""Alias para compatibilidad con 'python -m proyecto2'."""
from Proyecto_2.app import main
from Proyecto_2.ga_real import RealGAConfig, RealGeneticAlgorithm

__all__ = ["main", "RealGAConfig", "RealGeneticAlgorithm"]
