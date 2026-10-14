# Algoritmos Evolutivos en Python — Optimización de Funciones

Implementación completa de algoritmos genéticos para optimización de funciones matemáticas en Python sin librerías de terceros especializadas (desarrollado a mano, sin `deap`, `pygad`, ni similares), basada en la obra de referencia *Algoritmos Evolutivos: un enfoque práctico* (RA-MA).

---

## 1. Estructura del Repositorio

```
.
├── README.md                      # Documentación general y guía de usuario
├── requirements.txt               # Dependencias del proyecto
├── common/                        # Módulos compartidos por Proyecto 1 y Proyecto 2
│   ├── functions.py               # Catálogo de 12 funciones de prueba con óptimos teóricos
│   ├── stats.py                   # Dataclass GenerationStats y exportador a CSV
│   ├── study.py                   # Utilidades estadísticas para estudios paramétricos
│   └── ui/                        # Componentes reutilizables de interfaz
│       ├── theme.py               # Hoja de estilos moderna (tema oscuro y claro)
│       ├── plots.py               # Gráficas en tiempo real (PyQtGraph)
│       └── worker.py              # Hilo trabajador en segundo plano con control de animación
├── Proyecto_1/                    # PROYECTO 1: Algoritmo Genético Binario
│   ├── ga_binario.py              # Motor vectorizado del AG binario
│   ├── app.py                     # Interfaz de escritorio completa (PySide6)
│   ├── estudio.py                 # Script de análisis de sensibilidad paramétrica
│   ├── __main__.py                # Permite: python -m Proyecto_1
│   └── tests/                     # Pruebas unitarias completas (pytest)
│       ├── test_functions.py      # Validación de funciones y sus óptimos teóricos
│       ├── test_ga_binario.py     # Pruebas del motor binario y decodificación
│       └── test_ui.py             # Pruebas de integración de la interfaz
├── proyecto1/                     # Alias en minúsculas para 'python -m proyecto1'
├── Proyecto_2/                    # PROYECTO 2: Algoritmo Genético Real (siguiente entrega)
└── informe/                       # Resultados del estudio paramétrico (CSV, PNG, Markdown)
    ├── estudio_proyecto1.md       # Informe redactado con tablas y conclusiones
    ├── p1_comparativa_elitismo.png
    └── ...
```

---

## 2. Instalación y Entorno

Se recomienda Python 3.10 o superior (validado en Python 3.12).

```bash
# 1. Crear entorno virtual (opcional con paquetes del sistema si se desea)
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt
```

### Dependencias principales:
* `numpy` (≥ 1.26.0): cálculo numérico y vectorización de poblaciones.
* `PySide6-Essentials` (≥ 6.5.0): interfaz gráfica de escritorio moderna y fluida.
* `pyqtgraph` (≥ 0.13.0): trazado en tiempo real sin congelar la interfaz.
* `matplotlib` (≥ 3.6.0): exportación de figuras estadísticas para informes.
* `pandas` (≥ 2.0.0): tratamiento de tablas y resúmenes estadísticos.
* `pytest` (≥ 7.0.0): suite de pruebas automatizadas.

---

## 3. Uso del Proyecto 1 (AG Binario)

### 3.1 Ejecutar la Interfaz Gráfica (GUI)

La aplicación puede ejecutarse con cualquiera de los siguientes comandos equivalentes:

```bash
python -m Proyecto_1
# o bien:
python -m proyecto1
# o directamente:
python -m Proyecto_1.app
```

#### Características destacadas de la interfaz:
* **Tema oscuro por defecto** con paleta moderna (tonos Slate y azul cian) y botón para alternar a **tema claro**.
* **Selector de funciones categorizado**: Maximización 1D ($M_1 \dots M_4$), Minimización 1D ($m_1 \dots m_4$) y Multivariable ($V_1 \dots V_4$).
* **Fórmulas y óptimos visibles**: Cada función muestra su formulación matemática y el valor del óptimo teórico publicado en el libro.
* **Control interactivo de la animación**:
  * Control del retardo por generación con estimación de duración en tiempo real.
  * Presets de un clic: **10 Minutos (Exposición/Demo)**, **1 Minuto**, **10 Segundos** y **Máxima Velocidad**.
* **Controles de ejecución**: Iniciar, Pausar, Paso a Paso, Detener y Reiniciar.
* **Visualización en vivo (actualizada en cada ciclo)**:
  * **Gráfica A (Función y Población)**: Muestra la curva analítica continua, la distribución de toda la población en la generación actual, una estrella dorada sobre el mejor individuo y una línea discontinua roja en el óptimo teórico. En multivariable ($n > 2$), compara el vector de coordenadas del mejor contra el vector óptimo en un diagrama de barras.
  * **Gráfica B (Convergencia)**: Curvas simultáneas del mejor histórico, mejor de la generación, media de la población, peor individuo y línea del óptimo.
* **Exportación directa**: Botones para guardar las gráficas a imagen PNG y todo el historial de la evolución a CSV.

---

### 3.2 Ejecutar el Estudio de Parámetros por Línea de Comandos

El script ejecuta el protocolo experimental de variación de un parámetro a la vez fijando el resto en la configuración base del libro, más la comparativa con/sin elitismo:

```bash
# Ejecutar el estudio completo con N = 30 repeticiones por configuración:
python -m Proyecto_1.estudio --runs 30

# Modo rápido de verificación (menos generaciones y N = 5):
python -m Proyecto_1.estudio --quick
```

Los resultados se guardan automáticamente en la carpeta `informe/`:
* `informe/estudio_proyecto1.md`: Informe en Markdown con tablas y análisis detallado.
* `informe/p1_param_*.csv` y `informe/p1_param_*.png`: Archivos de datos y figuras para cada parámetro (`TOL`, `tam_pob`, `num_gen`, `prob_cruce`, `prob_mut`).
* `informe/p1_comparativa_elitismo.png`: Comparativa visual con y sin elitismo.

---

### 3.3 Ejecutar Pruebas Automatizadas

```bash
pytest Proyecto_1/tests/
```

La suite cubre:
* Verificación numérica exacta de las 12 funciones frente a sus óptimos conocidos.
* Decodificación binaria y verificación de que el gen 0 es el menos significativo (LSB).
* Garantía de que la transformación de minimización preserva aptitudes $> 0$ incluso con $c_{max} < 0$.
* Monotonía del mejor histórico bajo elitismo.
* Reproducibilidad estricta bajo la misma semilla aleatoria.
* Inicialización e integración de la interfaz offscreen.

---

## 4. Diferencias y Correcciones respecto al Código C++ Original del Libro

El código C++ del Apéndice B del libro presenta varias inconsistencias lógicas y errores de implementación. En esta traducción a Python se resolvieron meticulosamente manteniendo la fidelidad al marco conceptual:

1. **Factor de ajuste en minimización (1.05 vs 1.01)**:
   * *Código C++ original*: Empleaba `cmax + cmax / 100` (equivalente a un factor $1.01$).
   * *Corrección aplicada*: Se implementa el factor **$1.05$** tal como describe explícitamente el texto del libro, parametrizable en `BinaryGAConfig.cmax_factor`.

2. **Protección contra $c_{max} \le 0$ en minimización**:
   * *Código C++ original*: Si todos los valores de la función son negativos ($c_{max} < 0$, como en $m_2$, $m_3$, $m_4$ y $V_4$), la expresión `cmax * 1.05` resultaba en un valor *menor* que $c_{max}$, generando aptitudes transformadas negativas e invalidando la ruleta de probabilidades.
   * *Corrección aplicada*: Se utiliza `cmax + 0.05 * abs(cmax) + epsilon`. De esta forma, $c_{max\_ajustado}$ siempre supera estrictamente al valor máximo observado y todas las aptitudes transformadas resultan estrictamente positivas ($\ge \epsilon > 0$).

3. **Punto de cruce independiente por pareja y por variable**:
   * *Código C++ original*: Calculaba una única variable global `punto_cruce` una sola vez por generación y la reutilizaba para todas las parejas de padres.
   * *Corrección aplicada*: Para cada pareja seleccionada y para cada variable independiente, se muestrea un punto de corte aleatorio nuevo y fresco.

4. **Rango válido del punto de corte monopunto**:
   * *Código C++ original*: Podía generar `punto_cruce = lcrom`, produciendo hijos que no intercambiaban ningún gen (copias idénticas a los padres).
   * *Corrección aplicada*: El punto de corte se selecciona estrictamente en el intervalo $[1, lcrom - 1]$, garantizando que el cruce siempre recombine material genético de ambos progenitores.

5. **Condición de reemplazo de los descendientes**:
   * *Código C++ original*: Evaluaba `hijo.aptitud < adaptacion(padre) || !padre.elite`, produciendo inconsistencias al mezclar aptitud bruta con aptitud adaptada.
   * *Corrección aplicada*: Los hijos siempre reemplazan a los padres que no pertenecen a la élite. Los progenitores pertenecientes al 2 % élite se conservan intactos en la población siguiente.

6. **Inicialización del mejor individuo con soporte para valores negativos**:
   * *Código C++ original*: Inicializaba `aptitud_mejor = 0`, lo que provocaba que funciones con óptimos negativos nunca actualizaran el mejor encontrado.
   * *Corrección aplicada*: Se utiliza `np.argmax` / `np.argmin` directamente sobre las evaluaciones de la población, inicializando el registro histórico con el primer individuo evaluado.

7. **Precisión numérica de 64 bits en decodificación**:
   * *Código C++ original*: Utilizaba tipos `float` simples de 32 bits, perdiendo precisión en funciones con cromosomas largos ($lcrom > 23$).
   * *Corrección aplicada*: Cálculos vectorizados en `np.float64` asegurando precisión de hasta 16 dígitos significativos.

8. **Protección de la élite frente a la mutación**:
   * *Código C++ original*: El vector de élite no debía mutar, pero existían rutas en las que sus bits podían alterarse tras reevaluaciones.
   * *Corrección aplicada*: Los individuos élite se colocan en un bloque protegido contiguo (`0 .. tam_elite-1`) que queda estrictamente excluido tanto de la mutación por bit como del cruce monopunto.

9. **Registro histórico desacoplado (`best_so_far_raw`)**:
   * Se almacena tanto el mejor de la generación actual como el mejor histórico acumulado. Con elitismo activado, se garantiza formalmente que el mejor histórico es no decreciente (maximización) o no creciente (minimización).

10. **Vectorización y eliminación de pérdidas de memoria**:
    * La representación basada en arrays contiguos de NumPy evita las asignaciones dinámicas `new[]`/`delete[]` del C++, logrando tiempos de ejecución inferiores a 0.2 segundos para 1000 generaciones con 500 individuos.
