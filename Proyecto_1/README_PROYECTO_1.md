# Proyecto 1 — Optimización de funciones (A): Algoritmo Genético binario

> Lee antes `README_GENERAL.md` (restricciones, arquitectura, interfaz, estudio y entregables comunes).
> **Prohibido usar `deap`.** Referencia: libro «Algoritmos Evolutivos: un enfoque práctico», Proyecto 1 y Apéndice B (código C++).

## 1. Qué hay que hacer

1. **Traducir a Python** el código C++ del Apéndice B (módulos `funciones.h` / `funciones.cpp`) en sus tres variantes: **maximización**, **minimización** y **varias variables**.
2. **Graficar la evolución en cada generación** (con control de duración; preset de 10 minutos, ver `README_GENERAL.md` §6).
3. Construir una **interfaz atractiva** con parámetros interactivos.
4. Realizar el **estudio de parámetros** y la **comparativa con/sin elitismo**.

## 2. Diseño del algoritmo (opciones obligatorias del libro)

| Elemento | Decisión |
|---|---|
| Representación | Cadena binaria por variable (genotipo `bool`/`uint8`). Fenotipo = número real decodificado |
| Función de evaluación | Valor de la función en el punto decodificado (aptitud bruta) |
| Selección | **Ruleta** (puntuación relativa y acumulada) |
| Cruce | **Monopunto**: punto de corte aleatorio, se intercambian los segmentos |
| Mutación | **Bit a bit**: cada gen cambia con probabilidad `prob_mut` |
| Elitismo | **Obligatorio y activable/desactivable** (para la comparativa) |

### 2.1 Parámetros (línea de comandos del C++ → campos de la UI)

El C++ se invoca como `funciones x_min x_max TOL tam_pob num_gen prob_cruce prob_mut`. Ejemplo del libro para la primera función:

```
funciones 0.0 32.0 0.0001 500 1000 0.4 0.001
```

| Parámetro | Significado | Valor por defecto sugerido |
|---|---|---|
| `x_min`, `x_max` | Intervalo de búsqueda (uno por variable) | los de la función |
| `TOL` | Error permitido en la discretización del intervalo | `1e-4` |
| `tam_pob` | Tamaño de población | `500` |
| `num_gen` | Número límite de generaciones | `1000` |
| `prob_cruce` | Porcentaje/probabilidad de cruce | `0.4` |
| `prob_mut` | Probabilidad de mutación **por gen (bit)** | `0.001` |
| `elitismo` | Activar elitismo | `True` |
| `pct_elite` | % de la población que forma la élite | `2 %` (`TAM_ELITE = tam_pob*2/100`) |
| `semilla` | Semilla del RNG | aleatoria (visible) |

### 2.2 Fórmulas clave (idénticas al C++)

* **Longitud del cromosoma:** `lcrom = ceil( log2( 1 + (x_max − x_min) / TOL ) )` (una por variable).
* **Binario → decimal:** `bin_dec = Σ genes[i] · 2^i` (el gen 0 es el **menos significativo**, como en el C++).
* **Decodificación:** `x = x_min + (x_max − x_min) · bin_dec / (2^lcrom − 1)`.
* **Puntuación relativa:** `punt_i = aptitud_i / Σ aptitud`. **Acumulada:** suma acumulada de `punt` (en NumPy: `np.cumsum`; selección con `np.searchsorted(punt_acu, rng.random(tam_pob))`).

## 3. Funciones a implementar (todas, con su óptimo teórico)

> Verifica **cada una** con un test que la evalúe en su óptimo conocido. Los valores de esta tabla ya fueron comprobados numéricamente.

### 3.1 Maximización (1 variable)

| # | Función | Intervalo | Óptimo (libro) |
|---|---|---|---|
| M1 | `f(x) = sen⁶(5πx) · exp(−2·ln(2)·((x − 0.1)/0.8)²)` | [0, 1] | máx **1.0** en x = 0.1 |
| M2 | `f(x) = 20 + e − 20·exp(−0.2·|x|) − exp(cos(2πx))` | [0, 32] | máx **22.3136** en x = 31.5005 |
| M3 | `f(x) = x + |sen(32πx)|` | [0, 1] | máx **1.98442447** en x = 0.98447395 |
| M4 | `f(x) = 0.5 − (sen²(x) − 0.5) / (1 + 0.001·x⁴)` | [0, 10] | máx **1.0** en x = 0.0 |

### 3.2 Minimización (1 variable)

| # | Función | Intervalo | Óptimo (libro) |
|---|---|---|---|
| m1 | `f(x) = 1 + cos(x) / (1 + 0.01·x²)` | [0, 50] | mín **0.0883634** en x = 3.08531 |
| m2 | `f(x) = −|x · sen(√|x|)|` | [−250, 250] | mín **−201.843** en x = 203.814 (también en −203.814 por simetría) |
| m3 | `f(x) = sen(x) / (1 + √x + cos(x)/(1 + x))` | [0, 25] | mín **−0.318071** en x = 4.5798 |
| m4 | `f(x) = 5 + exp(0.5·sen(πx²)) − exp(0.1·x·cos(πx))` | [0, 25] | mín **−5.39204** en x = 23.9901 |

> ⚠️ **Nota sobre m4:** en el libro la fórmula se imprime de forma ambigua («exp(0.5(πx²))»). La versión correcta, comprobada numéricamente contra el mínimo publicado (−5.39204 en 23.9901), es la de la tabla (`0.5·sen(πx²)`). **No usar la lectura literal.**

### 3.3 Varias variables (n variables, cada una con su propia cadena de bits)

| # | Función | Dominio | Óptimo (libro) |
|---|---|---|---|
| V1 | `f(x₁,x₂) = 0.5 − (−0.5 + sen(√(x₁²+x₂²))²) / (1 + 0.001·(x₁²+x₂²))²` | x₁,x₂ ∈ [−40, 40] | **máx 1.0** en (0, 0) |
| V2 (Rastrigin) | `f(x) = 10n + Σᵢ (xᵢ² − 10·cos(2πxᵢ))` | xᵢ ∈ [−5.12, 5.12] | **mín 0** en x = 0 |
| V3 (Michalewicz, índice del libro) | `f(x) = −Σᵢ₌₁ⁿ sen(xᵢ) · sen²⁰( (i+1)·xᵢ² / π )` | xᵢ ∈ [0, π] | ver tabla ↓ |
| V4 | `f(x) = Σᵢ −xᵢ·sen(√|xᵢ|)` | xᵢ ∈ [0, 100] | **mín −n·63.63498** en xᵢ = 65.54785 |

Mínimos de V3 según `n` (verificados con la fórmula indicada, **con factor `(i+1)` e `i = 1..n`**):

| n | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mínimo | −1 | −1.959091 | −2.897553 | −3.886358 | −4.886358 | −5.879585 | −6.862457 | −7.858851 | −8.858851 | −9.856182 |

* `n` debe ser configurable en la UI para V2, V3 y V4 (V1 es fija con n = 2).
* Cada función del catálogo debe declarar: `nombre`, `fórmula (texto/LaTeX)`, `n_vars`, `limites`, `tipo ('max'|'min')`, `óptimo_valor`, `óptimo_x` y la propia función **vectorizada** (acepta arrays).

## 4. Algoritmo (pseudocódigo de referencia)

```
obtener_parametros()
lcrom[j] = ceil(log2(1 + (x_max[j] - x_min[j]) / TOL))         # una por variable
pob = poblacion_inicial()                                      # bits aleatorios 50/50
aptitud_bruta = evaluar(pob)                                   # g(x)
si minimiza: revisar_adaptacion_minimizar()                    # f = cmax' - g(x)
evaluacion()                                                   # sumaptitud, pos_mejor, puntuación y acumulada
yield estadísticas de la generación 0
para generacion en 1..num_gen:
    seleccion()          # élite (2 %) pasa directa; el resto por ruleta; población intermedia de tam_pob
    reproduccion()       # cruce monopunto sobre los elegidos con prob_cruce
    mutacion()           # bit-flip por gen con prob_mut (la élite NO muta)
    si minimiza: revisar_adaptacion_minimizar()
    evaluacion()
    yield estadísticas
devolver mejor individuo
```

### 4.1 Selección (`seleccion`)

1. Marcar como **élite** los `TAM_ELITE = tam_pob·2/100` individuos de mayor aptitud (si `elitismo=False`, no hay élite).
2. Para cada posición de la nueva población: si el individuo es élite, **sobrevive tal cual**; si no, se elige uno por **ruleta** (número aleatorio en [0,1) contra la puntuación acumulada).
3. Los elegidos forman la **población intermedia** que reemplaza a la anterior.

### 4.2 Reproducción / cruce (`reproduccion`, `cruce`)

1. Cada individuo se selecciona para cruce si `rand < prob_cruce`. Si el nº de seleccionados es impar, se descarta uno para que sea par.
2. Los seleccionados se emparejan consecutivamente (0-1, 2-3, …).
3. **Cruce monopunto:** hijo1 = padre1[:p] + padre2[p:], hijo2 = padre2[:p] + padre1[p:].
4. Los hijos **sustituyen a sus padres**, **excepto** que un padre de la élite no se sustituye (el elitismo conserva su genoma).
5. Con varias variables: se cruza **cada cadena `x_j` del padre 1 con la `x_j` del padre 2**, cada una con su propio punto de corte.

### 4.3 Mutación (`mutacion`)

Para cada individuo no élite y cada gen (bit): con probabilidad `prob_mut` se invierte (`!gen`). Si el individuo mutó, se **reevalúa**. Con varias variables, se aplica a **todas** las cadenas.

### 4.4 Minimización (`revisar_adap_mini`)

La función objetivo bruta `g(x)` se conserva en `aptitud_bruta`. La selección usa una aptitud transformada que es máxima cuando `g` es mínima:

```
cmax = max(aptitud_bruta de la población)
aptitud = cmax_ajustado − aptitud_bruta
```

Se **recalcula en cada generación** (después de mutación y antes de `evaluacion`). El resultado final y todas las gráficas usan `aptitud_bruta`.

## 5. Errores/rarezas del código C++ original (CORREGIR y documentar)

El agente debe traducir **la lógica**, no copiar los fallos. Corregir y listar en el README final:

1. **Factor 1.05 vs 1.01:** el texto del libro dice `cmax·1.05 − g(x)`, pero el código usa `cmax + cmax/100` (= 1.01). Usar **1.05** (parámetro configurable).
2. **`cmax` negativo:** si `cmax < 0` (p. ej. m2, m3, m4, V4), `cmax·1.05 < cmax` y las aptitudes salen **negativas** → ruleta inválida. Usar `cmax + 0.05·|cmax|` (equivale a ×1.05 si es positivo) y proteger el caso `cmax == 0` o población homogénea con un `ε` pequeño (o ruleta uniforme) para evitar división entre cero.
3. **Punto de cruce único por generación:** el C++ calcula `punto_cruce` **una sola vez** y lo usa para todas las parejas. Generar **uno nuevo por pareja** (y por variable).
4. **Rango del punto de corte:** el C++ puede dar `punto_cruce = lcrom` (no cruza nada). Usar enteros en `[1, lcrom−1]`.
5. **Condición de reemplazo de hijos:** el C++ usa `hijo.aptitud < adaptacion(padre) || !padre.elite`, que es incoherente (compara aptitudes y no actualiza `puntuación`). Implementar: *los hijos reemplazan siempre a los padres no élite; los padres élite se conservan*.
6. **Mejor individuo:** `aptitud_mejor` se inicializa en 0, lo que falla con aptitudes negativas. Inicializar con `−inf`/primer individuo (o usar `argmax`).
7. **Tipos `float` en cálculos de decodificación** (pérdida de precisión): usar `float64`.
8. **Fuga de memoria / `new[]`:** irrelevante en Python, no replicar.
9. **Mutación de la élite:** el C++ la protege; mantenerlo. Además, reevaluar siempre los individuos modificados (cruce y mutación).
10. Mantener un **registro del mejor histórico** (`best_so_far`) separado del mejor de la generación; con elitismo debe ser no decreciente (maximización) / no creciente (minimización).

## 6. Interfaz (específico del Proyecto 1)

Además de lo descrito en `README_GENERAL.md` §6:

* Selector de función agrupado: **Maximización / Minimización / Varias variables**; al elegirla se rellenan intervalo, tipo y óptimo teórico.
* Campos de parámetros: `x_min`, `x_max`, `TOL`, `tam_pob`, `num_gen`, `prob_cruce`, `prob_mut`, elitismo (switch), `% élite`, semilla, `n` (cuando aplique) y **duración de la animación**.
* Mostrar `lcrom` calculada (por variable) y el nº total de bits.
* **Gráfica A:** para 1 variable, curva de la función + población (puntos) + mejor (estrella) + óptimo teórico (línea/marca). Para 2 variables, contornos + población. Para n > 2, mostrar el mejor vector de variables (barras) y la convergencia.
* **Gráfica B:** mejor, media (y peor) por generación + línea del óptimo.
* Indicador «¿Alcanzó el óptimo? (|f − f*| < ε)» con color verde/rojo.

## 7. Estudio de parámetros (específico del Proyecto 1)

Seguir `README_GENERAL.md` §7. Parámetros a variar **uno a la vez** (los demás fijos en la configuración base del libro: `TOL=1e-4, tam_pob=500, num_gen=1000, prob_cruce=0.4, prob_mut=0.001`):

| Parámetro | Valores sugeridos |
|---|---|
| `TOL` (discretización) | 1e-1, 1e-2, 1e-3, 1e-4, 1e-5 |
| `tam_pob` | 20, 50, 100, 250, 500, 1000 |
| `num_gen` | 50, 100, 250, 500, 1000 |
| `prob_cruce` | 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 |
| `prob_mut` | 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1 |
| **Elitismo** | **Sí / No** (comparativa obligatoria) |

Aplicarlo como mínimo a **M2, M3 (multimodal), m2, V2 (n=5) y V3 (n=5)**. Reportar y comentar el efecto de cada parámetro y el nº de generaciones para llegar al óptimo con y sin elitismo.

## 8. Criterios de aceptación

- [ ] Las 12 funciones (4 máx, 4 mín, 4 multivariable) implementadas y verificadas contra sus óptimos con tests.
- [ ] El AG converge cerca del óptimo del libro con la configuración base (p. ej. M2 → ≈ 22.3136 en x ≈ 31.5005; m2 → ≈ −201.84).
- [ ] Las gráficas se actualizan **en cada generación**; el control de duración funciona (incluido el preset de 10 min) sin congelar la ventana.
- [ ] Pausa, paso a paso, reinicio y exportación (PNG/CSV) funcionan.
- [ ] Sin `deap` ni librerías evolutivas.
- [ ] Estudio de parámetros ejecutado y documentado, con comparativa con/sin elitismo.
- [ ] Lista de correcciones respecto al C++ incluida.
- [ ] `pytest` en verde.
