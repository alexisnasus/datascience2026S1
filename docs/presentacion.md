# Extracción de Presentación PDF

## --- SLIDE 1 / 21 ---

Predicción de
rendimiento SIMCE
mediante IDPS
GRUPO 1
Daniel Shinya
Alexis Lema
Sebastián Alonzo
Alexander Berríos
Bastian Lobos
Diego Escobar

## --- SLIDE 2 / 21 ---

El problema
Los Resultados SIMCE son el principal
indicador académico del país
Factores externos a lo academico, tales como
motivación o convivencia escolar pueden
influir en el rendimiento.
A día de hoy no existen modelos
predictivos de resultados académicos
que utilicen IDPS

## --- SLIDE 3 / 21 ---

¿Porqué?
Apoya la toma
Facilita la
Permite de decisiones
detección de
entender la políticas
establecimien
relación entre educativas
tos con riesgo
bienestar y basadas en
académico de
desempeño evidencia
forma
academico comprobable
temprana

## --- SLIDE 4 / 21 ---

Objetivo general
Desarrollar un modelo
de regresión capaz de
predecir puntajes
SIMCE utilizando
indicadores IDPS y
variables contextuales

## --- SLIDE 5 / 21 ---

Objetivos especificos
Evaluar la Identificar
Construit un
Analizar relación precisión del variables que
modelo
entre indicadores modelo tienen mayor
predictivo
IDPS y resultados mediante influencia en el
basado en
SIMCE métricas como rendimiento
regresión
RMSE y R² académico

## --- SLIDE 6 / 21 ---

Metodología basada en OSEMN
Enfoque en regresión supervisada
Variables predictoras: IDPS y contexto socioeconómico
Variable objetivo: Puntaje SIMCE
Enfoque
metodológico

## --- SLIDE 7 / 21 ---

Datos
Utilizados
Fuente:
Agencia de Calidad de la Educación
Resultados SIMCE
Indicadores de Desarrollo Personal y Social (IDPS).
Variables de vulnerabilidad y contexto escolar
Datos públicos oficiales de la agencia de calidad de la
educación

## --- SLIDE 8 / 21 ---

Preparación
de datos
Integración de bases SIMCE, IDPS y contexto escolar.
Eliminación de registros con valores faltantes críticos.
Codificación de variables categóricas relevantes.
Definición de variable objetivo y variables predictoras.

## --- SLIDE 9 / 21 ---

Preparación
de datos
Cobertura
• 9 años: 2016 – 2025 (excluyendo años de pandemia)
• 4 niveles evaluados: 4° básico, 6° básico, 8° básico y II medio
Desafíos detectados
• Formatos mixtos: .csv y .txt
• Tres delimitadores distintos: coma, pipe (|) y punto y coma (;)
• Encoding inconsistente: Latin-1 vs UTF-8
¿Por qué importa?
Sin detección automática de formato, un solo archivo mal leído
contamina años completos de datos y rompe la trazabilidad
histórica.

## --- SLIDE 10 / 21 ---

Drifts
estructurales
Cambio de formato impuesto por la Agencia
Antes (2016 – 2022): formato Wide
• Una columna por indicador IDPS (ind_am, ind_cc, ind_hv, ind_pf)
Desde 2023: formato Long
• Una sola columna "ind" con el nombre del indicador
• Una columna "prom" con el valor → filas derretidas por colegio
¿Por qué importa?
Sin homologar la estructura, el modelo vería las mismas variables
como columnas distintas y la matriz quedaría >80% en NaN,
inutilizable para regresión.

## --- SLIDE 11 / 21 ---

Limpieza
Consolidación
Nuestra variable a predecir: Puntajes SIMCE de Matemática y Lectura.
1. Limpieza del target
• Dropna estricto en prom_mate_rbd y prom_lect_rbd
• Sin Y no hay aprendizaje supervisado: filas sin puntaje
contaminan la validación cruzada y sesgan métricas
Posible justificación de reducción del dataset: La pérdida de registros corresponde a
establecimientos con notas en blanco o censuradas por secreto estadístico.
2. Estrategia de fusión
• Merge horizontal: une indicadores y sub-dimensiones por (rbd, año, curso)
• Append vertical: consolida toda la historia 2016 – 2025
• Deduplicación final por llave compuesta
3. Output: dataset maestro
• Datasets consolidados de IDPS y SIMCE listos para cruzar
• Diccionarios de datos en Markdown (idps + simce)
para gobernanza y reproducibilidad

## --- SLIDE 12 / 21 ---

1. Dataset IDPS consolidado
• 110.211 filas × 57 variables
• Cobertura: 8 años (2016, 2017, 2018, 2019, 2022, 2023, 2024, 2025)
• 4 niveles: 4° básico, 6° básico, 8° básico, II medio
Resultados de
• Incluye índices, sub-dimensiones y variables de contexto
la fase Obtain
2. Dataset SIMCE consolidado
• 53.657 filas × 7 variables
& Scrub • Cobertura: 5 años con SIMCE rendido (2018, 2019, 2023, 2024, 2025)
• Variables: rbd, año, puntajes mate/lectura, GSE, dependencia, ruralidad
Siguiente paso (en curso)
• Cruce IDPS × SIMCE por (rbd, año) para obtener el dataset analítico
• Reemplaza al dataset histórico reducido usado en la versión preliminar
del notebook, ampliando años y registros disponibles para el modelo
¿Por qué importa?
Tener las dos bases consolidadas y limpias por separado permite recombinarlas
con distintas estrategias (qué años incluir, cómo tratar nulos, etc) sin volver
a tocar los archivos crudos.

## --- SLIDE 13 / 21 ---

Estado Actual del Dataset (Versión Preliminar)
Alcance del
Para esta iteración, se construyó un dataset preliminar para
Dataset validar el pipeline (OSEMN) y los modelos base.
Preliminar
El foco específico es exclusivamente 2° Medio durante
Años 2017 a 2024.
Se consolidaron cientos de variables crudas en una matriz
final de exactamente 11 columnas clave, por ejemplo:
ind_am: Autoestima académica y motivación escolar.
ind_cc: Clima de convivencia escolar.
ind_hv: Hábitos de vida saludable.
ind_pf: Participación y formación ciudadana.
Se debe escalar este mismo cruce para integrar niveles de
básica (4°, 6°, 8° básico) y la totalidad del histórico de años,
expandiendo los registros.

## --- SLIDE 14 / 21 ---

Modelo
propuesto
Modelo principal: Regresión lineal múltiple
Evaluación mediante validación cruzada.
Interpretación de variables con feature importance

## --- SLIDE 15 / 21 ---

Alcance del
proyecto
Análisis a nivel de establecimientos
educacionales
Uso de datos históricos entre 2017 y 2025 (Excluyendo
los años en donde no se realizó por pandemia)
No busca establecer causalidad, sino relaciones
predictivas

## --- SLIDE 16 / 21 ---

Resultados
esperados
Resultado 1 Resultado 2 Resultado 3
identificar generar
Predecir el
factores clave
evidencia util
rendimiento
asociados a
académico con para
mejores
variables socio-
decisiones
emocionales resultados
educativas

## --- SLIDE 17 / 21 ---

Resultados preliminares

## --- SLIDE 18 / 21 ---

Resultados preliminares

## --- SLIDE 19 / 21 ---

Resultados preliminares
En matematica se obtuvo un R² de 0.592 indica que el modelo explica el 59.2% de la
variabilidad en los puntajes SIMCE entre establecimientos.
En lenguaje, se indica que el modelo explica el 55.7% de variabilidad.

## --- SLIDE 20 / 21 ---

El proyecto busca entregar un enfoque claro y
enfocado en regresión predictiva
La integración de IDPS y SIMCE puede aportar
información relevante para el sistema educativo chileno
Conclusión

## --- SLIDE 21 / 21 ---

Gracias

