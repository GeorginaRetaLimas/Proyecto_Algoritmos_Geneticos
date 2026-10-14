"""Alias para compatibilidad con 'python -m proyecto1'."""
from Proyecto_1.app import main
from Proyecto_1.ga_binario import BinaryGAConfig, BinaryGeneticAlgorithm

__all__ = ["main", "BinaryGAConfig", "BinaryGeneticAlgorithm"]
