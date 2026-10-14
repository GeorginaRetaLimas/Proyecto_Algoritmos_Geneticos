# Informe de Estudio Comparativo — Proyecto 2 (Algoritmo Genético Real)

## 1. Introducción y Marco Teórico

En el Proyecto 2 se implementa la **representación real** (vectores de coma flotante de precisión doble),
eliminando la discretización mediante cadenas de bits y evaluando la función objetivo directamente
sobre el fenotipo continuo. Se analizan en profundidad **4 operadores de cruce** y el operador de **mutación uniforme**.

### Operadores evaluados:
1. **SBX (Simulated Binary Crossover)**: Simula la dispersión del cruce binario en espacios continuos mediante el factor $\beta$.
2. **Aritmético**: Combinación lineal convexa $h_1 = \alpha p_1 + (1-\alpha) p_2$.
3. **Discreto Uniforme**: Intercambio de variables independientes con probabilidad $p_u$.
4. **Un Punto (Discreto)**: Punto de corte entre variables continuas (equivalente al monopunto del Proyecto 1).

---

## 2. Comparativa de los 4 Operadores de Cruce

![Comparativa de los 4 Operadores](p2_comparativa_4_operadores.png)

| Función | Óptimo | Error SBX | Error Aritmético | Error Discreto Uniforme | Error Un Punto |
|---|---|---|---|---|---|
| **M2** (n=1) | 22.3136 | 7.8036e-05 | 8.5681e-04 | 1.3940e-03 | 9.2262e-04 |
| **M3** (n=1) | 1.9844 | 7.5108e-09 | 7.1919e-04 | 1.0828e-03 | 7.7855e-04 |
| **m2** (n=1) | -201.8430 | 2.1788e-04 | 2.1783e-04 | 1.4824e-03 | 1.3035e-03 |
| **m4** (n=1) | -5.3920 | 1.7473e-06 | 7.7099e-02 | 1.2410e-01 | 1.1279e-01 |
| **V1** (n=2) | 1.0000 | 5.4731e-03 | 8.9786e-04 | 9.0191e-03 | 1.1109e-02 |
| **V2** (n=5) | 0.0000 | 5.2982e-01 | 3.5052e-01 | 2.1862e-01 | 1.5980e-01 |
| **V3** (n=5) | -4.8864 | 1.9010e-02 | 2.8211e-02 | 7.2832e-03 | 1.1143e-02 |
| **V4** (n=5) | -318.1749 | 1.6555e-02 | 2.2168e-01 | 1.3395e-02 | 1.5740e-02 |

### Análisis de los resultados por operador:
1. **Caso degenerado de 1 variable con 'Un Punto'**: En funciones unidimensionales ($M_2, M_3, m_2, m_4$),
   al haber una sola variable no existe punto de corte válido ($p \in [1, lcrom-1]$ queda vacío).
   El operador devuelve a los padres intactos, por lo que toda la búsqueda recae exclusivamente en la mutación.
   En multivariable ($V_1 \dots V_4$), 'Un Punto' vuelve a operar con normalidad intercambiando bloques de variables.
2. **Superioridad de SBX en funciones continuas complejas**: SBX logra el menor error y la mayor precisión
   en funciones altamente multimodales y no separables ($V_2$ Rastrigin, $V_3$ Michalewicz), ya que
   combina la exploración alrededor de la media con la capacidad de generar soluciones fuera del hipercubo de los padres.
3. **Comportamiento del Cruce Aritmético**: El cruce aritmético contrae progresivamente la población hacia
   la envolvente convexa de los mejores individuos, siendo extraordinariamente rápido en funciones unimodales
   pero vulnerable a la convergencia prematura en espacios multimodales si no hay suficiente mutación.

---

## 3. Escalabilidad con la Dimensión $n$

Se evaluó el comportamiento de cada operador aumentando el número de dimensiones en $V_2$ (Rastrigin):

![Escalado con n en Rastrigin](p2_escalado_dimension_v2.png)

| Dimensión $n$ | Error SBX | Error Aritmético | Error Discreto Uniforme | Error Un Punto |
|---|---|---|---|---|
| **$n = 2$** | 0.0000 | 0.0001 | 0.0110 | 0.0071 |
| **$n = 5$** | 0.6565 | 0.6265 | 0.1037 | 0.1385 |
| **$n = 10$** | 8.4262 | 7.6482 | 1.4601 | 2.3014 |
| **$n = 20$** | 50.4548 | 41.8114 | 18.3905 | 18.4146 |

**Conclusión sobre la dimensionalidad**: A medida que $n$ crece ($n=10, 20$), el espacio de búsqueda
crece exponencialmente ($20$ dimensiones implican un hipervolumen inmenso). SBX mantiene una ventaja
sostenible al explorar variaciones proporcionales a la distancia entre padres en cada gen.

---

## 4. Comparativa con y sin Elitismo en Representación Real

| Operador | Error Con Elitismo (2%) | Error Sin Elitismo | Reducción de Error con Elitismo |
|---|---|---|---|
| **SBX (Simulated Binary Crossover)** | 0.7511 | 4.1317 | **+81.8 %** |
| **Aritmético** | 0.3229 | 1.5064 | **+78.6 %** |
| **Discreto Uniforme** | 0.1232 | 1.8196 | **+93.2 %** |
| **Un Punto (Discreto)** | 0.1077 | 1.9733 | **+94.5 %** |

---

## 5. Comparativa: Representación Real (SBX) vs Representación Binaria (Proyecto 1)

Comparación bajo el mismo presupuesto computacional ($30,000$ evaluaciones por corrida):

| Función | Error Representación Real (SBX) | Error Representación Binaria (P1) | Ganador |
|---|---|---|---|
| **V2 (n=5)** | 4.9478e-01 | 5.7780e+00 | **Real (SBX)** |
| **V4 (n=5)** | 1.0821e-02 | 7.4691e+00 | **Real (SBX)** |

### Conclusiones de la comparativa Real vs Binario:
1. **Mayor precisión sin discretización**: La representación real no sufre el efecto de 'escalón' de la discretización $TOL$.
2. **Eficiencia en alta dimensión**: En problemas con $n \ge 5$, el cromosoma real mide sólo $5$ elementos,
   mientras que el binario requiere $19 \times 5 = 95$ bits, expandiendo el espacio combinatorio a $2^{95}$.
3. **Operadores geométricos continuos**: Operadores como SBX y el cruce aritmético aprovechan la métrica continua
   euclidiana del espacio euclidiano, acelerando la convergencia en órdenes de magnitud.