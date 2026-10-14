# Algoritmos Evolutivos en Python — Guía general (Proyectos 1 y 2)

> **Lee este archivo primero**, después `README_PROYECTO_1.md` y `README_PROYECTO_2.md`.
> Fuente: *Algoritmos Evolutivos: un enfoque práctico* (RA-MA), Parte II, Proyectos 1 y 2, y Apéndice B (código C++ del Proyecto 1).

---

## 1. Objetivo

Implementar en **Python** dos proyectos de optimización de funciones con algoritmos genéticos:

| Proyecto | Representación | Qué se entrega |
|---|---|---|
| **1** — Optimización de funciones (A) | **Binaria** (cadenas de bits) | Traducción fiel a Python del código C++ del libro + gráficas por generación + interfaz atractiva + estudio de parámetros |
| **2** — Optimización de funciones (B) | **Real** (vector de floats) | Mismo marco, con 4 operadores de cruce (un punto, uniforme, aritmético, SBX) + mutación uniforme + comparativa |

## 2. Restricciones obligatorias

1. **PROHIBIDO usar la librería `deap`.** Tampoco `pygad`, `pymoo`, `inspyred`, `geneticalgorithm` ni ninguna otra librería de algoritmos evolutivos. Todos los operadores (selección, cruce, mutación, elitismo) se programan a mano.
2. Se permite `numpy`, `scipy` (solo utilidades), `pandas` (tablas del estudio), `matplotlib`/`pyqtgraph` (gráficas) y la librería de interfaz elegida.
3. Python ≥ 3.10, con *type hints* y *docstrings* en español.
4. El motor del algoritmo **no debe importar nada de la interfaz** (separación estricta motor / UI).

## 3. Stack recomendado

```
numpy
PySide6          # interfaz de escritorio moderna
pyqtgraph        # gráficas rápidas, ideales para actualizar en cada generación
matplotlib       # solo para exportar figuras estáticas del informe
pandas
pytest
```

> Si el agente prefiere otra opción (p. ej. `customtkinter` + `matplotlib`, o `Streamlit`), puede usarla **solo si** cumple todos los requisitos de la sección 6 (en especial la actualización de gráficas en cada generación con control de velocidad). Debe justificarlo en el README final.

## 4. Estructura de repositorio sugerida

```
evolutivos/
├── README.md                    # instrucciones de instalación y uso (las escribe el agente)
├── requirements.txt
├── common/
│   ├── ui/                      # tema, widgets reutilizables (gráficas, panel de parámetros)
│   ├── functions.py             # catálogo de funciones objetivo (ver READMEs de cada proyecto)
│   ├── stats.py                 # dataclass GenerationStats + exportación CSV
│   └── study.py                 # utilidades de estudio paramétrico (multi-ejecución, tablas, gráficas)
├── proyecto1/
│   ├── ga_binario.py            # motor GA binario
│   ├── app.py                   # interfaz (python -m proyecto1)
│   ├── estudio.py               # script de estudio de parámetros
│   └── tests/
├── proyecto2/
│   ├── ga_real.py               # motor GA real
│   ├── operadores.py            # cruces y mutación
│   ├── app.py                   # interfaz (python -m proyecto2)
│   ├── estudio.py
│   └── tests/
└── informe/                     # resultados del estudio (CSV, PNG, informe .md)
```

## 5. Arquitectura del motor (aplica a ambos proyectos)

### 5.1 El algoritmo es un **generador**

Para poder graficar en cada generación sin acoplar la UI, el motor expone un generador:

```python
@dataclass
class GenerationStats:
    generation: int
    best_fitness_raw: float      # mejor valor REAL de la función (aptitud bruta, no la transformada)
    mean_fitness_raw: float
    worst_fitness_raw: float
    best_so_far_raw: float       # mejor histórico (monótono si hay elitismo)
    best_x: np.ndarray           # fenotipo del mejor (1 o n variables)
    population_x: np.ndarray     # (tam_pob, n_vars) fenotipos de TODA la población (para dibujarla sobre la función)
    population_fitness_raw: np.ndarray

class GeneticAlgorithm:
    def __init__(self, config, rng: np.random.Generator): ...
    def run(self) -> Iterator[GenerationStats]:
        """Yield tras la generación 0 (población inicial) y tras cada generación evolucionada."""
```

* La UI consume el generador desde un `QThread`/worker (o `QTimer`) y emite una señal por generación.
* Debe poder **pausar, reanudar, avanzar paso a paso, reiniciar y detener** sin bloquear la ventana.

### 5.2 Reproducibilidad

* Usar `np.random.default_rng(seed)` (no `random` global ni `rand()` estilo C). Semilla configurable en la UI y en los scripts; si está vacía, semilla aleatoria que se **muestra** para poder repetir la ejecución.

### 5.3 Cálculo vectorizado

* La población se guarda en arrays de NumPy (no listas de objetos) para que 500–1000 individuos × 1000 generaciones se ejecuten con fluidez. La traducción debe mantener la **lógica** del libro, no necesariamente la estructura de `struct`/`vector` de C++.

### 5.4 Minimización vs maximización

Ambos proyectos manejan maximización y minimización. El algoritmo trabaja internamente **siempre maximizando** una aptitud transformada (ver READMEs de cada proyecto), pero **todas las gráficas, tablas y resultados se muestran con el valor bruto de la función**.

## 6. Requisitos de interfaz (ambos proyectos)

La interfaz debe ser **atractiva y moderna**, no un formulario gris por defecto:

* **Tema oscuro por defecto** con opción de tema claro; paleta coherente (un color de acento), tipografía legible, espaciados generosos, esquinas redondeadas, iconos.
* **Layout** de tres zonas:
  1. **Panel izquierdo — Parámetros**: selector de función, intervalo, parámetros del AG (ver cada proyecto), semilla, interruptor de elitismo, botón **Ejecutar / Pausar / Paso / Reiniciar**.
  2. **Centro — Gráficas en vivo** (se actualizan **en cada generación**):
     * **Gráfica A – Función y población:** curva de la función (1 variable) o mapa de contornos (2 variables) con los individuos de la generación actual como puntos, el mejor resaltado y una marca del óptimo teórico conocido.
     * **Gráfica B – Convergencia:** aptitud **máxima/mejor, media** (y opcionalmente peor) por generación, más la línea del óptimo teórico.
  3. **Panel inferior/derecho — Resultados**: generación actual, mejor valor, mejor `x`, óptimo teórico, **error absoluto respecto al óptimo**, tiempo transcurrido, tabla/lista de las últimas generaciones.
* **Control de velocidad de animación (requisito clave):** el usuario debe ver la evolución **generación a generación**. Implementar:
  * Un control de *«Duración objetivo de la animación»* (segundos o minutos) que calcule el retardo por generación: `delay = duración / num_generaciones`.
  * Un **preset de 10 minutos** para la ejecución completa (modo demostración/exposición), además de presets más rápidos (p. ej. 10 s, 1 min) y **«Máxima velocidad»** (sin retardo; solo refresca la UI cada *k* generaciones).
  * El retardo debe aplicarse sin congelar la ventana ni acumular desfase (usar tiempo absoluto/`QTimer`, no `sleep` en el hilo de la UI).
* Botones para **exportar** la gráfica actual a PNG y el historial de la ejecución a CSV.
* Barra de estado con mensajes claros y validación de entradas (límites, tolerancia > 0, probabilidades en [0, 1], etc.) con mensajes de error amigables, nunca trazas de Python.
* Debe poder lanzarse con un solo comando (`python -m proyecto1`, `python -m proyecto2`).

## 7. Estudio de parámetros (ambos proyectos)

El libro exige un **estudio de resultados con distintos parámetros: se fijan todos menos uno y se varía ese**. Requisitos comunes:

* Cada configuración se repite **N = 30 ejecuciones** con semillas distintas (configurable) y se reporta **media ± desviación típica** (también mediana, mínimo y máximo).
* Métricas por configuración: mejor valor final (bruto), error respecto al óptimo teórico, **generación en la que se alcanza el óptimo dentro de una tolerancia** (o «no alcanzado»), % de ejecuciones exitosas y tiempo de CPU.
* Salida: tablas **CSV**, gráficas PNG (boxplots/barras con barras de error y curvas de convergencia medias) y un **informe `informe/estudio_proyectoX.md`** con tablas, figuras y **conclusiones redactadas** (qué valor de cada parámetro funciona mejor y por qué).
* **Comparativa con y sin elitismo** (obligatoria, el libro la pide explícitamente): mismos parámetros y semillas, midiendo calidad final y nº de generaciones hasta el óptimo.
* El estudio debe poder lanzarse por **línea de comandos** (sin UI) y, opcionalmente, desde una pestaña «Estudio» de la interfaz.

## 8. Calidad y pruebas

* `pytest` con, al menos: decodificación/codificación, longitud de cromosoma, que cada operador mantiene los genes dentro de rango, que el elitismo **nunca empeora** el mejor histórico, que la transformación de minimización mantiene aptitudes ≥ 0 y preserva el orden, valores de cada función en su óptimo conocido, y reproducibilidad con la misma semilla.
* Verificar cada función contra su **óptimo teórico** del libro (tablas en los READMEs de cada proyecto).
* Código formateado (`ruff`/`black`), sin números mágicos: parámetros por defecto en un único módulo de configuración.

## 9. Entregables finales

1. Código de ambos proyectos con la estructura anterior.
2. `README.md` raíz con instalación (`pip install -r requirements.txt`), uso, capturas de pantalla de la interfaz y explicación de decisiones de diseño.
3. Carpeta `informe/` con los resultados de los estudios (CSV + gráficas + informe en Markdown).
4. Tests pasando (`pytest`).
5. Un breve apartado «Diferencias respecto al código C++ del libro» listando cada corrección o mejora (ver sección «Errores/rarezas del código original» en `README_PROYECTO_1.md`).

## 10. Orden de trabajo sugerido

1. `common/functions.py` con **todas** las funciones y sus óptimos conocidos + tests de verificación.
2. Motor binario del Proyecto 1 (sin UI) + tests + comprobar que converge a los óptimos del libro.
3. Minimización y multivariable del Proyecto 1.
4. Interfaz del Proyecto 1 (gráficas en vivo + control de duración).
5. Estudio de parámetros del Proyecto 1.
6. Motor real del Proyecto 2 reutilizando selección/elitismo/transformación de minimización.
7. Operadores de cruce y mutación reales + tests.
8. Interfaz del Proyecto 2 (reutilizando `common/ui`).
9. Estudio comparativo del Proyecto 2.
10. Documentación, capturas e informe final.
