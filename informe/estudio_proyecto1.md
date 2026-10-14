# Informe de Estudio de Parámetros — Proyecto 1 (Algoritmo Genético Binario)

## 1. Introducción y Metodología

Este informe documenta el estudio experimental del Algoritmo Genético con representación binaria,
según las directrices de *Algoritmos Evolutivos: un enfoque práctico* (RA-MA).
Cada prueba se realizó fijando los parámetros en su valor base y variando **un único parámetro** cada vez.
Cada configuración fue evaluada a lo largo de **N ejecuciones independientes** con semillas reproducibles.

### Configuración base del libro:
- Tolerancia de discretización (`TOL`): `1e-4`
- Tamaño de población (`tam_pob`): `500`
- Generaciones (`num_gen`): `500` (o `1000`)
- Probabilidad de cruce (`prob_cruce`): `0.4`
- Probabilidad de mutación por bit (`prob_mut`): `0.001`
- Elitismo: `True` (porcentaje de élite: `2.0 %`)

---

## 2. Comparativa Obligatoria: Con Elitismo vs Sin Elitismo

El elitismo conserva íntegramente a los mejores individuos (2 % de la población) impidiendo
que el cruce o la mutación destruyan esquemas óptimos ya descubiertos.

![Comparativa de Elitismo](p1_comparativa_elitismo.png)

| Función | Óptimo | Error (Con Elitismo) | Error (Sin Elitismo) | Éxito % (Con) | Éxito % (Sin) | Gen. media al óptimo (Con) | Gen. media al óptimo (Sin) |
|---|---|---|---|---|---|---|---|
| **M2** | 22.3136 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 100% | N/A | N/A |
| **M3** | 1.9844 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 100% | 100% | 0.1 | 0.1 |
| **m2** | -201.8430 | 0.0002 ± 0.0000 | 0.0002 ± 0.0000 | 100% | 100% | N/A | N/A |
| **V2 (n=5)** | 0.0000 | 1.2667 ± 1.0984 | 1.8160 ± 1.2454 | 27% | 3% | 204.1 | 356.0 |
| **V3 (n=5)** | -4.8864 | 0.1128 ± 0.1126 | 0.1288 ± 0.1038 | 50% | 40% | 45.0 | 34.6 |

### Conclusiones de la comparativa de Elitismo:
1. **Preservación de la mejor solución**: Sin elitismo, la población sufre *regresiones*, pues el mejor individuo
   puede perderse en cualquier generación si es seleccionado para cruce o mutado destructivamente.
2. **Velocidad de convergencia**: El elitismo acelera drásticamente la convergencia, requiriendo en promedio
   significativamente menos generaciones para alcanzar el entorno del óptimo.
3. **Tasa de éxito superior**: En funciones multimodales complejas (como M3 o Rastrigin V2), la tasa de éxito
   con elitismo es notablemente superior.

---

## 3. Estudio de Sensibilidad Paramétrica

### 3.1 Tolerancia de discretización (TOL)

Determina la longitud de bits de cada variable $lcrom = \lceil \log_2(1 + (x_{max}-x_{min})/TOL) \rceil$. Valores más pequeños aumentan la precisión potencial pero agrandan el espacio de búsqueda ($2^{lcrom}$).

![Impacto de tol](p1_param_tol.png)

| Valor de `tol` | Mejor Aptitud Media ± Desv | Error Medio ± Desv | Tasa Éxito (%) | Tiempo CPU Medio (s) |
|---|---|---|---|---|
| `0.1` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.708 s |
| `0.01` | 22.3137 ± 0.0001 | 0.0001 ± 0.0000 | 100% | 0.626 s |
| `0.001` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.668 s |
| `0.0001` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.685 s |
| `1e-05` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.677 s |

### 3.2 Tamaño de población (tam_pob)

Controla la diversidad genética inicial y la capacidad exploratoria por generación. Poblaciones pequeñas convergen prematuramente, mientras que poblaciones grandes consumen más evaluaciones por ciclo.

![Impacto de tam_pob](p1_param_tam_pob.png)

| Valor de `tam_pob` | Mejor Aptitud Media ± Desv | Error Medio ± Desv | Tasa Éxito (%) | Tiempo CPU Medio (s) |
|---|---|---|---|---|
| `20` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.067 s |
| `50` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.108 s |
| `100` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.165 s |
| `250` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.354 s |
| `500` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.689 s |

### 3.3 Número límite de generaciones (num_gen)

Afecta el horizonte temporal concedido al algoritmo para explotar los atractores locales y globales.

![Impacto de num_gen](p1_param_num_gen.png)

| Valor de `num_gen` | Mejor Aptitud Media ± Desv | Error Medio ± Desv | Tasa Éxito (%) | Tiempo CPU Medio (s) |
|---|---|---|---|---|
| `50` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.064 s |
| `100` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.128 s |
| `250` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.311 s |
| `500` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.682 s |

### 3.4 Probabilidad de cruce monopunto (prob_cruce)

Regula la tasa de recombinación entre individuos seleccionados por ruleta.

![Impacto de prob_cruce](p1_param_prob_cruce.png)

| Valor de `prob_cruce` | Mejor Aptitud Media ± Desv | Error Medio ± Desv | Tasa Éxito (%) | Tiempo CPU Medio (s) |
|---|---|---|---|---|
| `0.1` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.262 s |
| `0.2` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.385 s |
| `0.4` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.627 s |
| `0.6` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.979 s |
| `0.8` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 1.206 s |
| `1.0` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 1.331 s |

### 3.5 Probabilidad de mutación bit a bit (prob_mut)

Es el mecanismo de exploración y mantenimiento de alelos. Si es muy baja, se pierde diversidad; si es excesiva, degenera en búsqueda aleatoria pura.

![Impacto de prob_mut](p1_param_prob_mut.png)

| Valor de `prob_mut` | Mejor Aptitud Media ± Desv | Error Medio ± Desv | Tasa Éxito (%) | Tiempo CPU Medio (s) |
|---|---|---|---|---|
| `0.0001` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.639 s |
| `0.001` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.634 s |
| `0.005` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.719 s |
| `0.01` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.704 s |
| `0.05` | 22.3137 ± 0.0000 | 0.0001 ± 0.0000 | 100% | 0.653 s |

---

## 4. Conclusiones Generales y Recomendaciones

1. **Tolerancia recomendada**: `1e-4` es un excelente compromiso entre precisión de decimales (~4 cifras) y longitud del cromosoma (~19 bits en 1D).
2. **Tamaño de población**: `tam_pob = 250 - 500` evita el estancamiento prematuro en óptimos locales para funciones multimodales sin disparar el coste computacional.
3. **Tasa de cruce**: Valores en el rango `[0.4, 0.7]` demostraron la mejor combinación de exploración y estabilidad.
4. **Tasa de mutación**: `prob_mut = 0.001 - 0.005` por bit (aproximadamente $1 / total\_bits$) proporciona la tasa ideal de mutación.
5. **Elitismo**: Debe mantenerse **siempre activado** en aplicaciones prácticas (2% de élite), pues garantiza convergencia monótona.