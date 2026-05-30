# Roadmap de Modelamiento: Predicción SIMCE vía IDPS (2016-2025)

Este documento sirve como la guía técnica de referencia y especificación de requisitos para la fase de **Explore & Model** del proyecto, resolviendo las brechas identificadas por la pauta de evaluación del profesor y las notas de retroalimentación en vivo.

---

## 1. Redefinición Arquitectónica y Objetivos
* **Re-enfoque Estratégico:** El objetivo general deja de ser "construir un modelo de regresión lineal" (que constituye una herramienta metodológica) y pasa a ser **"anticipar y diagnosticar el desempeño académico (SIMCE) de los establecimientos educativos a partir de variables contextuales e indicadores socioemocionales (IDPS) previos, para permitir la intervención temprana en recintos con riesgo escolar"**.
* **Dualidad de Targets:** No se promediarán los puntajes. Se entrenarán **dos modelos completamente independientes** debido a que la literatura y la varianza de los datos demuestran que los factores socioemocionales impactan de forma diferenciada:
    1.  `Target_Mat`: `prom_mate2m_rbd` (Rendimiento en Matemática)
    2.  `Target_Lec`: `prom_lect2m_rbd` (Rendimiento en Lectura)

---

## 2. Ingeniería de Variables Basada en Teoría Científica (Clases 05 y 07)

### A. Codificación Estricta del Nivel Socioeconómico (NSE)
* **Problema previo:** Tratar `cod_grupo` (NSE) mediante One-Hot Encoding destruye la relación de orden natural de la variable (Bajo < Medio Bajo < Medio < Medio Alto < Alto).
* **Solución:** Implementar un **Ordinal Encoding** explícito asignando valores discretos secuenciales (ej. `1` para Bajo, `2` para Medio Bajo, ..., `5` para Alto). Esto permite que los coeficientes del modelo lineal capturen adecuadamente el efecto incremental del contexto socioeconómico.

### B. Mitigación de Multicolinealidad en Variables IDPS
* **Diagnóstico:** Las predictoras de desarrollo personal (`ind_am`, `ind_cc`, `ind_hv`, `ind_pf`) provienen de encuestas institucionales correlacionadas entre sí. Una alta multicolinealidad desestabiliza los coeficientes $\beta$ de la regresión OLS.
* **Acción Requerida:** 1.  Calcular la **Matriz de Correlación de Pearson** entre todos los predictores continuos.
    2.  Calcular el **Factor de Inflación de la Varianza (VIF)** para cada variable. Todo predictor con un $\text{VIF} > 5$ debe ser evaluado con cautela.
    3.  Si la colinealidad es severa, implementar **Regresión Regularizada (Ridge, Lasso o Elastic Net)** (Clase 16) en lugar de OLS básico, para penalizar los pesos magnánimos y estabilizar la variabilidad.

### C. Análisis de Simetría y Transformaciones
* **Acción Requerida:** Graficar histogramas de las variables predictoras continuas y de los targets para evaluar el sesgo (*skewness*). Si las distribuciones muestran colas largas o asimetrías severas, aplicar transformaciones matemáticas (Logarítmica, Raíz Cuadrada o Box-Cox) antes del escalamiento con `StandardScaler` para asegurar el cumplimiento del supuesto de normalidad en los errores del modelo predictivo.

---

## 3. Protocolo de Partición Temporal y Validación (Clases 11, 12 y 15)

### A. Prevención de Fuga de Información (Information Leakage)
* **Contexto Temporal:** Dado que el dataset consolida series históricas por establecimiento (`rbd`) entre los años 2016 y 2025, un train/test split completamente aleatorio (`train_test_split`) causará que el modelo entrene con datos de un colegio en el año $T+1$ para predecir el año $T$, invalidando la capacidad predictiva futura real.
* **Estrategia de Partición:** Implementar una **Partición Temporal Excluyente (Out-of-Time Validation)**:
    * **Train Set:** Registros correspondientes a los años de entrenamiento histórico (ej. 2016 a 2023).
    * **Test Set:** Reservar exclusivamente los años más recientes (ej. 2024 y/o 2025) como conjunto de prueba ciego. El modelo jamás debe ver estos años durante el ajuste o la selección de hiperparámetros.

### B. Desfase Temporal Predictivo (Temporal Lag)
* Para que el modelo posea utilidad práctica institucional, las variables IDPS deben registrarse en un año previo ($T-1$) al año del target SIMCE ($T$). Estructurar el dataset final desplazando los vectores de características IDPS temporalmente hacia atrás respecto a las variables objetivo.

---

## 4. Métricas de Performance y Diagnósticos (Clases 11, 15 y 18)

### A. Justificación Rigorosa del MAE
* Se seleccionará y defenderá el **MAE (Error Absoluto Medio)** como métrica de negocio principal debido a su alta interpretabilidad para los *stakeholders* educativos: expresa la desviación del modelo directamente en **puntos reales de la escala SIMCE**.
* Se reportará simultáneamente **RMSE** (para penalizar penalizaciones cuadráticas por errores grandes) y **$R^2$** (para detallar la proporción de la varianza explicada).

### B. Monitoreo Train vs. Test
* Es obligatorio generar salidas en tablas impresas y curvas de aprendizaje (*Learning Curves*) que comparen el rendimiento entre el conjunto de Train y el conjunto de Test. Una brecha significativa donde $\text{MAE}_{\text{train}} \ll \text{MAE}_{\text{test}}$ alertará inmediatamente de sobreajuste (*Overfitting*).

### C. Análisis de Residuos e Influencia
Tras ajustar el estimador lineal, se deberán ejecutar de manera mandatoria dos análisis estadísticos:
1.  **Residual Plot:** Gráfico de dispersión de los residuos ($e_i = y_i - \hat{y}_i$) vs. valores predichos para verificar el supuesto de **homocedasticidad** (varianza constante del error).
2.  **Distancia de Cook:** Calcular la influencia individual de cada establecimiento en los coeficientes. Toda observación que supere el umbral crítico establecido por la regla:
    $$D_i > 4 \times \text{mean}(D)$$
    debe ser etiquetada como *Influential Point*, investigando si corresponde a un error de digitalización en las bases de datos originales de la Agencia de Calidad o a un caso atípico justificado que deba ser tratado.
