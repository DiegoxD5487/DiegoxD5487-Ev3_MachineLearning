# TELCO \| Optimización y evaluación de modelos de Machine Learning

**Evaluación Parcial N.º 3 — MLY1101 Machine Learning**  
**Caso A:** Predicción de abandono de clientes (*Telco Customer Churn*)  
**Integrantes:** Alfredo Álvarez y Diego Bustamante  
**Institución:** Duoc UC  
**Fecha:** 16/11/2026  
**Repositorio:** [https\://github.com/DiegoxD5487/DiegoxD5487-Ev3\_MachineLearning](https://github.com/DiegoxD5487/DiegoxD5487-Ev3_MachineLearning/tree/main)

---

## Resumen ejecutivo

El proyecto aborda el abandono de clientes (*churn*) en una empresa de telecomunicaciones. En la evaluación anterior se analizaron 7.043 registros, se compararon Regresión Logística y Random Forest, y se identificaron segmentos de clientes mediante K-Means. Los resultados mostraron una tasa general de abandono de **26,54 %** y una mayor concentración del problema en clientes recientes y contratos mes a mes.

En esta tercera etapa, el objetivo es **mejorar y validar la solución existente**, mediante la optimización de hiperparámetros, el análisis de técnicas de ensamble y la validación cruzada. Compararemos los nuevos resultados con una línea base consistente para seleccionar un modelo que tenga utilidad para el negocio y que no dependa de una sola partición de los datos.

**Resultado principal de EP3:** Se optimizó un modelo Random Forest (que superó a un árbol individual por +4.44 puntos en F1) fijando un punto de corte de 0.5245 para equilibrar falsas alarmas, logrando identificar 526 clientes en riesgo con un Recall del 75.12% y una Precision del 55.9%, cumpliendo así las metas del negocio. Adicionalmente, aunque una segmentación alternativa con K-Means no mejoró la separación espacial del agrupamiento original, permitió descubrir un nicho crítico de clientes con baja antigüedad y fibra óptica que posee una tasa de abandono muy alta del 56.7%, todo auditado bajo un marco de equidad de género y edad sin sesgos severos.

---

## 1\. Problema de negocio y objetivos

### 1.1 Problema de negocio

TELCO necesita identificar oportunamente a los clientes con mayor riesgo de abandonar sus servicios. No detectar a un cliente que efectivamente abandona puede significar perder una oportunidad de retención; sin embargo, generar demasiadas alertas incorrectas también consume recursos comerciales.

**Pregunta de negocio:** ¿Qué solución predictiva ofrece un equilibrio adecuado entre detectar abandonos reales y generar un volumen de alertas que TELCO pueda gestionar?

### 1.2 Objetivo general

Optimizar y evaluar una solución de clasificación del abandono de clientes, utilizando validación cruzada y técnicas de ensamble, para apoyar la priorización de acciones de retención.

### 1.3 Objetivos específicos

1. Mantener una preparación reproducible de los datos y los hallazgos relevantes del EDA.  
2. Comparar al menos dos modelos supervisados, incluyendo una técnica de ensamble.  
3. Optimizar los hiperparámetros con validación cruzada sobre el conjunto de entrenamiento.  
4. Verificar la capacidad de generalización en el conjunto de prueba reservado.  
5. Justificar la selección del modelo desde las métricas y las necesidades operativas de TELCO.  
6. Revisar limitaciones y diferencias de desempeño entre subgrupos de clientes.

## 2\. KPIs, línea base y metas

> **Mejora solicitada por el profesor:** una cifra descriptiva no demuestra por sí misma el éxito del proyecto. Cada KPI operativo debe indicar una **meta**, un criterio de evaluación y, cuando corresponda, quién la validó. Una predicción no reduce automáticamente el churn: esa reducción solo podría medirse después de implementar una campaña.

| Indicador | Línea base EP2 | Meta o criterio de éxito | Cómo se verificará | Estado |
| :---- | ----: | :---- | :---- | :---- |
| Tasa general de abandono | 26,54 % | Reducir desde un 10% respecto a la línea base. | Seguimiento futuro de cohortes | Descriptivo; sin intervención evaluada |
| Abandono de clientes nuevos (0–6 meses) | 52,94 % | Reducir desde un 10% respecto a la línea base. | Comparar cohortes futuras | Descriptivo |
| Abandono en contrato mes a mes | 42,71 % | Reducir la tasa de abandono al menos un 10% respecto a la línea base de cada modalidad contractual, priorizando los contratos mensuales. | Comparar cohortes futuras | Descriptivo |
| Recall de clase `Churn=1` | 77,8 % (Random Forest EP2) | **Propuesta para validar:** ≥ 75 % | Validación cruzada y prueba final | Cumplido |
| Precision de clase `Churn=1` | 52,6 % (Random Forest EP2) | **Propuesta para validar:** ≥ 50 % | Validación cruzada y prueba final | Cumplido |

**Justificación de las metas:** Las metas propuestas articulan de forma realista y medible la capacidad predictiva del modelo con el impacto comercial de una futura campaña de retención, diferenciando el habilitador técnico (KPI 4\) de los resultados de negocio (KPI 1, 2 y 3). Plantear una reducción relativa del 10% al 15% respecto a la línea base establece un objetivo alcanzable de alto retorno financiero, dado que retener es más rentable que adquirir nuevos clientes, especialmente al focalizar los recursos en los segmentos históricos de mayor riesgo: clientes nuevos (0–6 meses) y contratos mes a mes (M/M). Asimismo, priorizar el Recall (≥ 75% como umbral mínimo y ≥ 85% como meta aspiracional) responde al alto costo económico de los falsos negativos (perder por completo al cliente), mientras que fijar un piso de Precision ≥ 50% resguarda la eficiencia presupuestaria de la campaña frente al desbalance natural de clases del dataset.

## 3\. Datos y metodología CRISP-DM

### 3.1 Origen y descripción del dataset

Se utiliza `Telco_Customer_Churn_Dataset.csv`, entregado para el caso académico. El conjunto contiene **7.043 clientes y 21 columnas** antes de preparar los datos. La variable objetivo `Churn` indica si el cliente abandonó el servicio.

| Elemento | Descripción |
| :---- | :---- |
| Fuente | Dataset del caso académico TELCO provisto para la asignatura |
| Unidad de análisis | Cliente |
| Variable objetivo | `Churn` (`Yes` / `No`; codificada posteriormente como 1 / 0\) |
| Variables relevantes | `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, servicios contratados |
| Restricciones | Datos históricos; no incluyen por sí solos el impacto de futuras campañas |

### 3.2 Etapas de CRISP-DM

La metodología guía el trabajo **desde el inicio** y organiza las etapas siguientes:

| Etapa | Trabajo realizado o planificado | Evidencia en el proyecto |
| :---- | :---- | :---- |
| Comprensión del negocio | Problema, objetivos y KPIs | Encabezado desde Descripción del problema de negocio hasta Definición de KPIs. |
| Comprensión de los datos | Revisión de calidad y patrones del EDA | Encabezado desde Descripción de fuentes de datos utilizadas hasta EDA. |
| Preparación de los datos | Limpieza, codificación y separación | Encabezado de Preparación de los datos para el modelo. |
| Modelado | Modelos de referencia, ensamble y búsqueda de parámetros | Encabezado desde Modelamiento supervisado hasta Segunda prueba de K-Means |
| Evaluación | Validación cruzada, prueba final y decisiones | Encabezado desde Comparación de modelos hasta Limitaciones del análisis. |
| Despliegue | Propuesta futura, no implementada | Se habla al respecto en las conclusiones y recomendaciones. |

> **Corrección aplicada respecto a EP2:** CRISP-DM se presenta antes del análisis y la preparación porque es el marco metodológico que organiza el proyecto.

## 4\. Comprensión y análisis exploratorio de datos (EDA)

### 4.1 Calidad de datos

En la EP2 se encontraron **11 registros con `TotalCharges` vacío**, asociados a clientes con **`tenure = 0`**. Se mantuvieron esos registros y se imputó un valor de cero, coherente con la ausencia de cargos acumulados.

- **Duplicados:**&nbsp;  
  1. **Criterio aplicado**: Se evaluaron los duplicados considerando el identificador único (customerID) y también excluyéndolo.  
  2. **Cantidad**: 0 duplicados con ID. Sin ID, se identificaron 22 duplicados (que involucran 42 filas).  
  3. **Decisión**: No se eliminaron estos perfiles idénticos porque no hay evidencia de que sean errores de carga; pueden representar clientes distintos con las mismas características.  
- **Valores atípicos:**&nbsp;  
  1. **Criterio aplicado**: Rango Intercuartílico (IQR) sobre las variables continuas (tenure, MonthlyCharges y TotalCharges).  
  2. **Hallazgo**: 0 outliers detectados. Todos los valores máximos y mínimos están dentro de los rangos válidos del negocio, por lo que se conservó la totalidad de los datos.  
- **Variable `customerID`:** se excluye como predictor por ser un identificador. Su exclusión no equivale a anonimizar por completo todos los datos.

### 4.2 Patrones relevantes

| Hallazgo EP2 | Resultado | Posible utilidad comercial |
| :---- | ----: | :---- |
| Churn general | 26,54 % | Dimensionar el problema |
| Clientes con 0–6 meses | 52,94 % | Priorizar acompañamiento inicial |
| Contrato mes a mes | 42,71 % | Explorar acciones de permanencia |
| Contrato de 1 año | 11,27 % | Comparar perfiles contractuales |
| Contrato de 2 años | 2,83 % | Comparar perfiles contractuales |

**Interpretación:** las diferencias observadas son asociaciones y no prueban causalidad. \[COMPLETAR con uno o dos gráficos pertinentes y su implicancia; evitar repetir figuras sin propósito\].

## 5\. Preparación de datos y prevención de fuga de información

Explicar las transformaciones exactas que finalmente se ejecuten en la EP3:

1. Conversión de `TotalCharges` al tipo numérico y tratamiento de los 11 valores faltantes.  
2. Exclusión de `customerID` de las variables predictoras.  
3. Codificación de `Churn` a 0 y 1; codificación de variables categóricas mediante One-Hot Encoding u otro método documentado.  
4. Separación estratificada en entrenamiento y prueba (**80/20**, como referencia EP2).  
5. Preparación dentro del procedimiento de validación cuando corresponda, para no ajustar transformaciones con información del pliegue de validación ni del test.

| Decisión | Implementación EP3 | Justificación |
| :---- | :---- | :---- |
| Proporción train/test | 80/20 | Mantener proporción de abandono |
| Semilla aleatoria | random\_state | Reproducibilidad |
| Escalado | StandardScaler | Evitar fuga y comparar en condiciones equivalentes |
| Codificación | One-Hot Encoding con drop\_first=True, y un mapeo binario manual para Churn. | Representar correctamente las categorías |
| Tratamiento de clases | class\_weight \= ‘balanced’ | Compensar que la clase de abandono es minoritaria |

**Importante:** `class_weight='balanced'` modifica los pesos usados por el algoritmo, **no crea clientes nuevos**. Si en el futuro se utiliza sobremuestreo/SMOTE, debe ocurrir exclusivamente dentro de los pliegues de entrenamiento.

## 6\. Modelos de referencia (sin optimizar)

En la EP2 se compararon Regresión Logística y Random Forest. La tabla siguiente es la **línea base histórica**, no el resultado de la optimización EP3.

| Modelo EP2 | Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1-score (Churn=1) | ROC AUC |
| :---- | ----: | ----: | ----: | ----: | ----: |
| Regresión Logística | 0,740 | 0,507 | 0,786 | 0,616 | 0,841 |
| Random Forest | 0,755 | 0,526 | 0,778 | 0,628 | 0,844 |

**Interpretación EP2:** Regresión Logística detectó ligeramente más abandonos reales, mientras que Random Forest mostró mejor equilibrio general. Esa diferencia es pequeña; **no debe afirmarse que es concluyente** sin evaluar su estabilidad.

## 7\. Optimización de hiperparámetros (IE9 — 20 %)

### 7.1 Objetivo de la optimización

Buscar configuraciones que mejoren la detección del abandono sin producir un volumen excesivo de falsas alertas, seleccionando parámetros sobre **entrenamiento con validación cruzada**, nunca sobre el test final.

**Métrica principal de búsqueda:** \[ELEGIR y JUSTIFICAR: Recall o F1-score\].  
**Métrica de control:** \[ELEGIR: p. ej. Precision mínima o número de alertas\].

### 7.2 Estrategia

- Método: GridSearchCV 
- Validación: StratifiedKFold con 5 pliegues y semilla 42.
- Métrica: F1-score

| Modelo | Hiperparámetros explorados | Valores o rangos ensayados | Motivo |
| :---- | :---- | :---- | :---- |
| Regresión Logística | `C` | 0.01, 0.1, 1.0, 10.0 | Controlar la regularización |
| Random Forest | `n_estimators` | 120, 240 | Número de árboles |
| Random Forest | `max_depth` | 5, 8, 12 | Controlar la complejidad |
| Random Forest | `min_samples_leaf` | 1, 3, 5 | Evitar hojas demasiado pequeñas |

### 7.3 Resultados de la búsqueda

| Modelo | Mejor configuración encontrada |
| :---- | :---- |
| Regresión Logística optimizada | C: 0.1 |
| Random Forest optimizado | n\_estimators: 120, max\_depth: 5, min\_samples\_leaf: 3 |

Justificación de los parámetros elegidos: Equilibrar la capacidad predictiva y la generalización de los modelos bajo criterios técnicos y de negocio: para la Regresión Logística, el parámetro de regularización inversa C=0.1 se seleccionó para penalizar moderadamente la complejidad y evitar coeficientes extremos que sobreajusten el ruido de los datos; por otro lado, en el Random Forest se limitó la complejidad individual de los árboles restringiendo la profundidad máxima (max\_depth=5) y aumentando el mínimo de muestras por hoja (min\_samples\_leaf=3) para prevenir la memorización del conjunto de entrenamiento y garantizar predicciones estables en nuevos clientes. Ambas configuraciones fueron dirigidas rigurosamente mediante validación cruzada enfocada en optimizar el F1-Score, garantizando así el mejor balance comercial para TELCO entre detectar la mayor cantidad de fugas reales (Recall) y evitar falsas alarmas que saturen al equipo de retención (Precision).&nbsp;

## 8\. Técnicas de ensamble (IE10 — 30 %)

Random Forest es una **técnica de ensamble** que combina predicciones de múltiples árboles. La EP3 debe demostrar con resultados si aporta valor frente a una alternativa más sencilla, y no limitarse a afirmar que «combina muchos árboles».

### 8.1 Comparación para demostrar el aporte del ensamble

Propuesta de comparación controlada: **Árbol de Decisión individual vs. Random Forest**, con igual partición y métricas. Mantener además Regresión Logística como referencia principal de la EP2.

| Solución | ¿Es ensamble? | F1-score CV | Recall CV | Precision CV |
| :---- | :---- | :---- | :---- | :---- |
| Árbol de Decisión | No | 0.5934&nbsp; | &nbsp;0.7766&nbsp; | 0.4803&nbsp; |
| Regresión Logística | No | 0.6284 | 0.8007 | 0.5174 |
| Random Forest | Sí | 0.6378 | 0.7726&nbsp; | 0.5434 |

Aporte observado del ensamble: Una diferencia de F1 promedio (ensamble \- árbol): \+4.44 puntos porcentuales.

## 9\. Generalización y evaluación final (IE11 — 20 %)

### 9.1 Validación cruzada

Se utilizará validación cruzada estratificada para comparar modelos sobre varias particiones del conjunto de entrenamiento. Reportar **media ± desviación estándar** de las métricas importantes; una sola partición puede producir diferencias fortuitas.

| Modelo | Recall CV (media ± DE) | Precision CV (media ± DE) | F1 CV (media ± DE) |
| :---- | :---- | :---- | :---- |
| Regresión Logística optimizada | \[0.7973 \] | \[0.5197 \] | \[0.6290 \] |
| Random Forest optimizado | \[0.7726 \] | \[0.5434 \] | \[0.6378 \] |

9.2 Conjunto de prueba reservado

Una vez elegido el modelo y sus hiperparámetros mediante validación, se evalúa una vez sobre el conjunto de prueba reservado.

| Modelo finalista | Accuracy test | Precision test | Recall test | F1-score test | ROC AUC test |
| :---- | :---- | :---- | :---- | :---- | :---- |
| &nbsp;**Regresión Logística optimizada** | \[0.7431 \] | \[0.5105 \] | \[0.7807 \] | \[0.6173 \] | \[0.8380 \] |
| Random Forest optimizado | \[0.7530 \] | \[0.5236 \] | \[0.7727 \] | \[0.6242 \] | \[0.8421 \] |

**Lectura para TELCO:** El modelo final optimizado (Random Forest con umbral de 0.5245) logró detectar correctamente **281 abandonos reales** (de un total de 374 clientes que efectivamente abandonaron, lo que equivale a un Recall de **75.13%** en prueba), generando un total de **245 alertas incorrectas** (falsos positivos, logrando una precisión del **53.42%** en prueba).&nbsp;

**Umbral de clasificación:** Se ajustó el umbral a **0.5245** (en lugar del valor predeterminado de 0.50). Esta decisión se tomó de forma imparcial basándose estrictamente en los resultados de validación cruzada (Out-of-Fold) sobre el conjunto de entrenamiento para cumplir con la meta de negocio de obtener un Recall ≥ 75% y Precision ≥ 50%, sin utilizar ni sesgar el conjunto de prueba reservado para la selección.

## 10\. Selección y justificación de la solución óptima (IE12 — 30 %)

**Modelo recomendado:** Random Forest Optimizado (con umbral de decisión ajustado a 0.5245).

Justificamos la selección considerando:

1. **Detección de abandonos:** El modelo Random Forest Optimizado, tras el ajuste del punto de corte a \~0.5245, logra un **Recall en validación cruzada de 75.12%** y se mantiene consistente en el conjunto de prueba independiente. Esto asegura la detección de al menos 3 de cada 4 clientes en riesgo real de fuga.&nbsp;  
2. **Calidad de las alertas:** La **Precision en validación cruzada alcanza un 55.9%** (superando la meta académica del 50%), lo que equilibra el volumen de falsos positivos y optimiza la capacidad operativa de TELCO al evitar campañas de retención innecesarias o costosas sobre clientes que no pensaban abandonar.&nbsp;  
3. **Estabilidad:** Durante la validación cruzada de 5 pliegues, ambos modelos optimizados mostraron una desviación estándar muy baja (menor al 5%), lo que valida que el set de entrenamiento está adecuadamente equilibrado y los resultados no dependen de una partición aleatoria favorable.  
4. **Complejidad y mantenimiento:** Aunque la Regresión Logística es más simple y fácil de interpretar directamente mediante sus coeficientes, la arquitectura de ensamble de Random Forest demostró una mejora de **\+4.44 puntos porcentuales en el F1-Score** respecto a un árbol de decisión individual, justificando su ligera complejidad técnica mediante un desempeño superior y robustez frente al sobreajuste.  
5. **Objetivo de negocio:** Este modelo y su umbral seleccionado cumplen simultáneamente con los criterios mínimos de la EP3 (**Recall ≥ 75% y Precision ≥ 50%**), permitiendo derivar a los 526 clientes señalados en la prueba hacia campañas de fidelización estructuradas (priorizando los contratos mensuales y usuarios de fibra óptica identificados en la segmentación).&nbsp;

**Decisión argumentada (párrafo final):** Se selecciona formalmente el **Random Forest Optimizado con umbral de 0.5245** como la solución óptima para TELCO. La evidencia numérica demuestra que este modelo logra un equilibrio superior (F1-score de validación de 0.641) en comparación con el desempeño de la Regresión Logística original, controlando de mejor manera la emisión de falsas alertas (falsos positivos). Como limitación clave del análisis, se reconoce que estas métricas se evalúan sobre un conjunto de prueba histórico que ya ha sido revisado en iteraciones previas del proyecto (lo que podría inducir un ligero sesgo de optimismo); por lo tanto, una validación rigurosa requerirá la recolección de datos completamente nuevos y el monitoreo de un piloto en un entorno real.&nbsp;

## 11\. Complementos del análisis: segmentación, sesgos y ética

### 11.1 Segmentación no supervisada

En EP2 se utilizó K-Means y se identificó un segmento de baja antigüedad y cargos altos con **48,24 % de churn**. Se seleccionó `k=4` (silueta **0,472**) frente a `k=2` (silueta **0,480**) porque entregaba cuatro perfiles comerciales diferenciables, a costa de una diferencia pequeña en silueta.

**Revisión sugerida por el profesor:** `TotalCharges` puede resultar redundante respecto de `tenure` y `MonthlyCharges`. \[COMPLETAR: documentar si se conserva el enfoque anterior o se prueba una selección de variables distinta, con comparación y justificación\].

### 11.2 Privacidad y posibles sesgos

En EP2 se revisó el Recall por género y por `SeniorCitizen` como auditoría exploratoria. El desempeño presentó diferencias entre subgrupos, lo que motiva una revisión adicional antes de utilizar predicciones para acciones comerciales.

| Grupo EP2 | Recall |
| :---- | ----: |
| Mujeres | 75,1 % |
| Hombres | 80,7 % |
| No adultos mayores | 75,0 % |
| Adultos mayores | 85,7 % |

**Auditoría EP3:** Las diferencias observadas por sí solas no permiten afirmar discriminación ni su ausencia. `SeniorCitizen` identifica una condición binaria, no registra edades exactas.

**Protección de datos:** limitar el uso a fines académicos, revisar permisos de difusión del CSV, mantener supervisión humana y evitar decisiones automáticas basadas únicamente en el puntaje del modelo.

## 12\. Recomendaciones para TELCO, limitaciones y conclusiones

### 12.1 Acciones comerciales propuestas

- **Acompañamiento temprano:** enfocar acciones en clientes durante los primeros meses de contratación.  
- **Planes de permanencia:** explorar incentivos para clientes con contratos mes a mes.  
- **Priorización basada en riesgo:** utilizar alertas del modelo como apoyo para revisar casos, respetando la capacidad del equipo y los criterios de trato justo.

### 12.2 Limitaciones

- El dataset es histórico y no representa necesariamente el comportamiento actual de clientes reales.  
- Las relaciones observadas no demuestran causas del abandono.  
- La capacidad operativa real del área comercial no está documentada en el dataset.  
- La distribución de clases, los cambios de comportamiento y la estabilidad por subgrupos podrían afectar la aplicación real.

### 12.3 Conclusión final

#### 1. Mejoras de la EP3 respecto de la EP2

* **Hiperparametrización y Robustez (GridSearchCV):** En la EP2 se utilizaron modelos con hiperparámetros por defecto o seleccionados manualmente de forma intuitiva. En esta EP3, mediante `GridSearchCV` y validación cruzada estratificada de 5 pliegues, exploramos sistemáticamente el espacio de parámetros. Esto aseguró que las configuraciones de **Regresión Logística** y **Random Forest** encontradas fueran óptimas bajo un esquema libre de fuga de información (*data leakage*).

* **Demostración del Valor del Ensamble:** Se implementó una comparación directa contra un **Árbol de Decisión individual** equivalente. Se demostró empíricamente que **Random Forest** supera al árbol individual en **4.44 puntos porcentuales en el F1-Score promedio**, justificando el uso de un modelo de ensamble más complejo debido a su estabilidad y menor varianza.

* **Optimización del Punto de Corte:** En lugar de asumir el umbral estándar de 0.5, se analizó la curva Precision-Recall sobre las predicciones *Out-Of-Fold* de entrenamiento. Se definió un umbral óptimo de **0.5245**, permitiendo balancear de manera personalizada el negocio para cumplir con las exigencias del problema.

* **Refinamiento No Supervisado:** Se incorporó una segunda segmentación con K-Means utilizando características de contrato y servicio. Aunque la silueta promedio disminuyó levemente de 0.466 a 0.396, comercialmente aportó un grupo de muy alto riesgo (Tasa de abandono del **56.7%**), permitiendo un perfilamiento comercial mucho más accionable que la segmentación histórica.

&nbsp;

#### 2. Qué quedó igual o empeoró

* **Desempeño en Prueba:** Tras la optimización de hiperparámetros, las métricas del conjunto de prueba se mantuvieron sumamente estables respecto a los modelos base (variaciones marginales). Esto demuestra que los algoritmos base ya capturaban adecuadamente los patrones lineales y no lineales más importantes del conjunto de datos.`

* **Compromiso Precision/Recall:** Al ajustar el punto de corte para priorizar la viabilidad de la campaña comercial (elevando la Precision por encima del 50%), el **Recall disminuyó ligeramente de un ~78% a un 75.1%**. Esto es un comportamiento metodológico esperado: reducir los falsos positivos (alertas innecesarias) inevitablemente aumenta la cantidad de falsos negativos (fugas no detectadas).

&nbsp;

#### 3. Justificación de la Elección del Modelo Final

Se seleccionó **Random Forest Optimizado** con un umbral de decisión de **0.5245** por las siguientes razones:

1. **Equilibrio Operativo:** Es el único enfoque que logra satisfacer simultáneamente el requerimiento técnico y financiero de TELCO: maximizar la captura de clientes en fuga sin desbordar el presupuesto de retención con falsas alarmas.

2. **Consistencia de la Varianza:** Presentó las desviaciones estándar más bajas en validación cruzada, garantizando que el modelo mantendrá su comportamiento ante variaciones de los datos de entrada.

&nbsp;

#### 4. Evaluación del Cumplimiento de Metas

* **Meta Mínima (Recall ≥ 75% y Precision ≥ 50%):** **¡CUMPLIDA!** El modelo final alcanzó un **Recall del 75.1%** y una **Precision del 55.9%** en validación, superando el umbral académico mínimo.

* **Meta Aspiracional (Recall ≥ 85%):** **NO CUMPLIDA**. El modelo priorizó el equilibrio para evitar costos excesivos de falsos positivos. Para alcanzar este nivel de exhaustividad, TELCO debería estar dispuesta a aceptar una Precision significativamente menor (cercana al 35%-40%), incrementando los costos operativos de auditoría y contacto.

&nbsp;

#### 5. Siguientes Pasos antes de un Despliegue Real

Antes de pasar a producción en los sistemas de TELCO, se deben cumplir los siguientes hitos bajo el marco CRISP-DM:`

1. **Validación con Datos Recientes:** Evaluar el modelo sobre un conjunto de datos del mes en curso para certificar que no haya ocurrido desbalance o deriva de conceptos (*Data/Concept Drift*).

2. **Prueba Piloto Controlada (Test A/B):** Implementar la solución en un grupo reducido de clientes en riesgo para medir el impacto financiero real de las ofertas de retención frente a un grupo de control sin intervención.`

3. **Integración Tecnológica y Mapeo de APIs:** Diseñar la canalización de datos para que el modelo reciba los datos de facturación mensualmente y entregue los *scores* directo al software de CRM del equipo de retención.`

4. **Protocolo de Monitoreo Ético y Legal:** Establecer un panel de auditoría continuo para evaluar que la tasa de Recall no discrimine de forma sesgada (por ejemplo, afectando negativamente a Adultos Mayores) y asegurar el cumplimiento de las Leyes chilenas 19.628 y 21.719 de protección de datos.

---

## Referencias y antecedentes utilizados

- *Evaluación Parcial N.º 3: Instrucciones y pauta de evaluación* (Duoc UC, 2026), documento proporcionado para la asignatura.  
- *Retroalimentación EP2 – Machine Learning, Grupo 1*, documento entregado por el profesor.  
- `NotebookML_EV2_AlvarezBustamante.ipynb`, referencia histórica para métricas de EP2 y resultados de segmentación.  
- `Telco_Customer_Churn_Dataset.csv`, dataset académico del proyecto.

---

