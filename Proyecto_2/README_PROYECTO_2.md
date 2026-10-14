# Proyecto 2 — Optimización de funciones (B): Algoritmo Genético con representación real

> Lee antes `README_GENERAL.md` y `README_PROYECTO_1.md`. Este proyecto es una **ampliación del Proyecto 1**: misma selección por ruleta, mismo elitismo, misma transformación para minimizar y mismo catálogo de funciones, pero con **cromosomas reales**.
> **Prohibido usar `deap`.** El libro **no incluye código** para este proyecto: se implementa a partir de la descripción y los pseudocódigos de este documento.

## 1. Qué hay que hacer

1. Implementar un AG con **representación real** (cada gen = una variable en coma flotante).
2. Implementar **4 operadores de cruce** (un punto, discreto uniforme, aritmético, SBX) y **mutación uniforme**.
3. **Graficar la evolución en cada generación** (preset de 10 minutos, ver `README_GENERAL.md` §6) con interfaz atractiva.
4. Estudio comparativo de operadores y parámetros, con más profundidad en **funciones de varias variables** (el libro lo pide expresamente: «se han probado más a fondo las funciones de varias variables»).

## 2. Diseño del algoritmo

| Elemento | Decisión |
|---|---|
| Representación | Vector de reales; **cada valor es un gen** = una variable |
| Función de evaluación | Evaluar la función directamente sobre el vector (no hay decodificación) |
| Selección | **Ruleta** (igual que Proyecto 1) |
| Elitismo | Igual que Proyecto 1: 2 % de la población, activable |
| Cruce | **Un punto**, **discreto uniforme**, **aritmético** y **SBX** (seleccionable en la UI) |
| Mutación | **Uniforme**: el gen se reemplaza por un valor aleatorio dentro de sus límites |

Ejemplo de cromosoma de 5 variables:

```
genes = [3.24, 2.71, 0.87, 2.34, 4.55]     # x1..x5
```

Los individuos de 1 variable tienen cromosoma de **longitud 1**.

### 2.1 Parámetros

| Parámetro | Valor por defecto sugerido |
|---|---|
| Función objetivo | selector (catálogo completo) |
| `n` (nº de variables, cuando aplique) | 2 … 30 (por defecto 5) |
| `tam_pob` | 100 |
| `num_gen` | 500 |
| `prob_cruce` | 0.8 |
| `prob_mut` | 0.05 (**por gen**) |
| Operador de cruce | SBX |
| `p_u` (cruce uniforme) | 0.5 |
| `alfa` (cruce aritmético) | 0.5 (opción: «aleatorio por gen») |
| `n_sbx` (índice de distribución SBX) | 1 (valor del libro) |
| Elitismo / % élite | Sí / 2 % |
| Semilla | aleatoria (visible) |

## 3. Funciones (mismo catálogo del Proyecto 1)

**Reutilizar `common/functions.py`** del Proyecto 1: las 4 de maximización, las 4 de minimización y las 4 multivariable (V1 Schaffer-like, V2 Rastrigin, V3 Michalewicz, V4 Σ −x·sen√|x|), con **los mismos óptimos** indicados en `README_PROYECTO_1.md` §3 (incluida la corrección de la función m4: `5 + exp(0.5·sen(πx²)) − exp(0.1·x·cos(πx))`).

Con representación real, los límites de cada gen son `[inf_i, sup_i]` (pueden ser distintos por variable). Valores de referencia que el libro cita en este proyecto:

* **Rastrigin** `f(x) = 10n + Σ (xᵢ² − 10 cos(2πxᵢ))`, xᵢ ∈ [−5.12, 5.12]: mínimo 0. Evaluación del libro: `suma += x² − 10·cos(2π·x); return 10·n + suma`.
* **Σ −x·sen(√|x|)**, xᵢ ∈ [0, 100]: mínimo `−n·63.63498` en xᵢ = 65.54785. Evaluación del libro: `suma −= x·sen(√|x|)`.
* **Michalewicz** (tabla de mínimos por `n` en `README_PROYECTO_1.md`).

## 4. Algoritmo

El bucle es el mismo del Proyecto 1 (selección → reproducción → mutación → revisar adaptación si se minimiza → evaluación), cambiando la representación y los operadores. La **selección y la transformación de minimización se reutilizan** (con las mismas correcciones: factor 1.05 sobre `cmax + 0.05·|cmax|`, protección contra aptitudes negativas y división por cero; ver §5 de `README_PROYECTO_1.md`).

### 4.1 Inicialización

```
genera_indiv():
    para i en 0..lcrom-1:
        genes[i] = aleatorio_real(limite_inf[i], limite_sup[i])
```

### 4.2 Cruce — reglas comunes

* Se eligen individuos para cruzar con `prob_cruce`, se hace par el nº de seleccionados y se emparejan consecutivamente (igual que Proyecto 1).
* Los hijos **sustituyen a los padres no élite**; la élite se conserva.
* **Todos los hijos deben quedar dentro de `[inf_i, sup_i]`**: aplicar *clipping* (o reparación) tras el cruce, especialmente en aritmético y SBX. Documentar la decisión.
* Reevaluar siempre los hijos.

### 4.3 Operadores de cruce

**a) Un punto (discreto, igual que el Proyecto 1 pero sobre genes reales).** Punto de corte aleatorio `p ∈ [1, lcrom−1]` nuevo para cada pareja; hijo1 = padre1[:p] + padre2[p:], hijo2 = padre2[:p] + padre1[p:].
> ⚠️ **Caso límite:** con funciones de **1 variable** (`lcrom = 1`) no existe punto de corte válido y este cruce **no hace nada**. Debe manejarse sin errores, avisar en la UI y reflejarlo en el estudio (no es un fallo: es una limitación real del operador).

**b) Discreto uniforme.** Probabilidad `p_u`. Para cada gen `i`:
```
si alea() < p_u:  hijo1[i] = padre2[i];  hijo2[i] = padre1[i]      # intercambia
si no:            hijo1[i] = padre1[i];  hijo2[i] = padre2[i]      # hereda
```

**c) Aritmético.** `0 ≤ α ≤ 1` (constante, o aleatorio por gen):
```
hijo1[i] = α·padre1[i] + (1 − α)·padre2[i]
hijo2[i] = α·padre2[i] + (1 − α)·padre1[i]
```

**d) SBX (Simulated Binary Crossover), con n = 1.** Se genera `α = alea()` y:
```
si α < 0.5:   β = (2·α)^(1/(n+1))
si α ≥ 0.5:   β = (1 / (2·(1 − α)))^(1/(n+1))

hijo1[i] = 0.5 · ( (padre1[i] + padre2[i]) − β·|padre2[i] − padre1[i]| )
hijo2[i] = 0.5 · ( (padre1[i] + padre2[i]) + β·|padre2[i] − padre1[i]| )
```
Decidir y documentar si `α` se genera **una vez por pareja** (como en el pseudocódigo del libro) o **por gen**; ofrecer la opción en configuración (por defecto: una vez por pareja, como el libro).

### 4.4 Mutación uniforme

```
para cada individuo no élite:
    para cada gen j:
        si alea() < prob_mut:
            genes[j] = aleatorio_real(inf[j], sup[j])
            mutado = cierto
    si mutado: reevaluar
```
Los límites `inf/sup` son **los de ese gen**. La élite no muta.

## 5. Interfaz (específico del Proyecto 2)

Además de `README_GENERAL.md` §6 (reutilizar `common/ui`):

* **Selector de operador de cruce** (Un punto / Uniforme / Aritmético / SBX) con sus parámetros específicos (`p_u`, `α`, `n_sbx`) que se muestran/ocultan según el operador.
* **Modo «Comparar operadores»:** ejecuta los 4 operadores **en paralelo con la misma semilla/población inicial** y superpone en la gráfica de convergencia las 4 curvas (mejor y/o media) con leyenda; una tabla muestra el resultado final de cada uno. Se anima generación a generación con el mismo control de duración (preset 10 min).
* **Selector de `n`** para V2, V3 y V4.
* **Gráfica A:**
  * 1 variable: curva + población + mejor + óptimo.
  * 2 variables: mapa de contornos + población (mostrar cómo se concentra o dispersa según el operador).
  * n > 2: **gráfico de barras o coordenadas paralelas** del mejor cromosoma frente al óptimo conocido (cuando existe, p. ej. xᵢ = 65.54785 en V4), más el **error medio por gen**.
* **Gráfica B:** mejor, media (y peor) por generación + óptimo; con la opción de mostrar la **diversidad de la población** (desviación típica media de los genes), útil para explicar la convergencia prematura.
* Aviso visible si se elige «Un punto» con funciones de 1 variable.

## 6. Estudio comparativo (específico del Proyecto 2)

Seguir `README_GENERAL.md` §7 (30 ejecuciones por configuración, media ± desv., CSV + gráficas + informe).

1. **Comparativa de los 4 cruces** sobre: M2, M3, m2, m4 (1 variable; recordar el caso degenerado del cruce de un punto) y **V1, V2, V3, V4** con `n ∈ {2, 5, 10, 20}` (V1 solo n = 2).
2. **Variar un parámetro cada vez** (resto fijo): `tam_pob`, `num_gen`, `prob_cruce`, `prob_mut`, y los parámetros propios de cada operador (`p_u` ∈ {0.1, 0.3, 0.5, 0.7, 0.9}; `α` ∈ {0.1, 0.25, 0.5, 0.75, 0.9} y aleatorio; `n_sbx` ∈ {0.5, 1, 2, 5}).
3. **Con vs sin elitismo** para cada operador.
4. **Escalado con `n`:** gráfica de error final frente a `n` para cada operador (muestra cuál soporta mejor la dimensionalidad).
5. **(Extra recomendado)** Comparar contra la **representación binaria del Proyecto 1** en las mismas funciones multivariable (si el módulo del Proyecto 1 está disponible), a igual nº de evaluaciones.
6. El informe debe concluir: qué operador es mejor según el tipo de función (unimodal/multimodal, separable o no), cómo influye `n`, y el efecto del elitismo y de la diversidad.

## 7. Pruebas mínimas

* Cada cruce devuelve genes dentro de `[inf, sup]`; longitudes correctas; no modifica a los padres (copias).
* Aritmético con α = 0.5 produce dos hijos idénticos (media de los padres); con α = 1 devuelve los padres intercambiados correctamente (comprobar contra la fórmula).
* SBX: `hijo1 ≤ hijo2` componente a componente (con β ≥ 0) y su media coincide con la media de los padres.
* Un punto con `lcrom = 1` no lanza excepción y devuelve los padres sin cambios.
* Mutación uniforme con `prob_mut = 0` no cambia nada; con `prob_mut = 1` cambia todos los genes no élite.
* Elitismo: el mejor histórico nunca empeora.
* Minimización: aptitudes transformadas ≥ 0 y orden preservado, también con `cmax < 0`.
* Reproducibilidad con semilla fija.

## 8. Criterios de aceptación

- [ ] Representación real y los **4 cruces** + mutación uniforme implementados a mano (sin `deap`).
- [ ] Todas las funciones del catálogo funcionan con `n` configurable donde aplique y alcanzan valores cercanos a su óptimo del libro.
- [ ] Gráficas actualizadas **en cada generación**, con control de duración (preset de 10 min) y modo «Comparar operadores».
- [ ] Interfaz atractiva y consistente con la del Proyecto 1 (tema oscuro/claro, componentes compartidos).
- [ ] Caso de 1 variable + cruce de un punto manejado y comunicado.
- [ ] Estudio completo (cruces, parámetros, elitismo, escalado con `n`) con CSV, gráficas e informe con conclusiones.
- [ ] `pytest` en verde.
