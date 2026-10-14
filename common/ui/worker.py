"""Worker en segundo plano (QThread) para ejecutar el generador del AG con control interactivo."""

from __future__ import annotations

import time
from typing import Iterator

from PySide6 import QtCore

from common.stats import GenerationStats


class GeneticAlgorithmWorker(QtCore.QThread):
    """Hilo trabajador para ejecutar el motor del AG de forma desacoplada de la interfaz."""

    generation_updated = QtCore.Signal(GenerationStats)
    evolution_finished = QtCore.Signal()
    status_changed = QtCore.Signal(str)
    error_occurred = QtCore.Signal(str)

    def __init__(self, ga_generator: Iterator[GenerationStats], delay_seconds: float = 0.05, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.generator = ga_generator
        self.delay_seconds = max(0.0, float(delay_seconds))

        self._mutex = QtCore.QMutex()
        self._pause_cond = QtCore.QWaitCondition()

        self._is_paused = False
        self._is_stopped = False
        self._step_once = False
        self._max_speed = (self.delay_seconds == 0.0)

    def set_delay(self, seconds: float) -> None:
        """Actualiza el retardo deseado entre generaciones."""
        self._mutex.lock()
        self.delay_seconds = max(0.0, float(seconds))
        self._max_speed = (self.delay_seconds == 0.0)
        self._mutex.unlock()

    def pause(self) -> None:
        """Pausa la evolución."""
        self._mutex.lock()
        self._is_paused = True
        self._mutex.unlock()
        self.status_changed.emit("Pausado")

    def resume(self) -> None:
        """Reanuda la evolución."""
        self._mutex.lock()
        self._is_paused = False
        self._step_once = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()
        self.status_changed.emit("Ejecutando")

    def step(self) -> None:
        """Avanza exactamente una generación y se pausa."""
        self._mutex.lock()
        self._step_once = True
        self._is_paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()

    def stop(self) -> None:
        """Detiene la evolución completamente."""
        self._mutex.lock()
        self._is_stopped = True
        self._is_paused = False
        self._pause_cond.wakeAll()
        self._mutex.unlock()

    def run(self) -> None:
        """Bucle de ejecución del hilo."""
        try:
            gen_count = 0
            for stats in self.generator:
                # Comprobar si se solicitó detener
                self._mutex.lock()
                if self._is_stopped:
                    self._mutex.unlock()
                    break

                # Manejar pausa y paso a paso
                while self._is_paused:
                    self._pause_cond.wait(self._mutex)
                    if self._is_stopped:
                        break

                if self._is_stopped:
                    self._mutex.unlock()
                    break

                if self._step_once:
                    self._is_paused = True
                    self._step_once = False
                    self.status_changed.emit("Paso completado (Pausado)")

                delay = self.delay_seconds
                max_speed = self._max_speed
                self._mutex.unlock()

                # Emitir señal a la UI
                # En modo máxima velocidad se puede omitir refrescar algunas generaciones intermedias si van muy rápido
                if max_speed and stats.generation > 0:
                    gen_count += 1
                    # Refrescar cada 5 generaciones o si es la última
                    if gen_count % 5 == 0:
                        self.generation_updated.emit(stats)
                else:
                    self.generation_updated.emit(stats)

                # Aplicar retardo si no es velocidad máxima
                if delay > 0.0:
                    time.sleep(delay)

            # Emitir la última generación siempre
            if not self._is_stopped and 'stats' in locals():
                self.generation_updated.emit(stats)

            self.evolution_finished.emit()

        except Exception as e:
            self.error_occurred.emit(str(e))
