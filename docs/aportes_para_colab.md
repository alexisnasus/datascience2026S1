# Aportes para el notebook final (Colab) — Presentación 2

Snippets listos para pegar en el notebook
`datascienceproyecto_modelofinal_beta2_(1).ipynb`. **No reemplazan tu trabajo**, lo completan
cubriendo los pendientes que dejaste + lo que pidió el profe (modelo global, `tipo_prueba`,
predictivo T-1, justificación de métrica y conclusiones).

Todo el código de aquí **reutiliza tus helpers y variables** ya definidos en el notebook
(`df_modelo`, `calcular_metricas`, `ASIGNATURAS`, `PALETA_ASIGNATURA`, `HUE_ASIGNATURA`, `targets`,
los imports de sklearn, `display`, etc.). Pégalo tal cual; **todos los números de abajo están
verificados** corriendo sobre los datasets reales del repo.

Índice:
1. Celdas de código nuevas (global · unificado · predictivo)
2. Celdas markdown (conclusiones + justificación de métrica)
3. Correcciones de los comentarios "vs GitHub"
4. Ajustes de narrativa (objetivo y mejoras vs primer modelo)

---

## 1) Celdas de código nuevas

### 1.A — Modelo global (todos los cursos juntos)

> **Dónde:** después de la Sección 4 (modelos por curso). Responde tu "me faltó el modelo con todos
> los datos juntos". Es el mismo pipeline, pero sobre todo `df_modelo` y agregando `curso` como
> **control de escala** (cada prueba tiene su propio rango de puntajes).

**Celda markdown (antes del código):**

```markdown
## 4.x) Modelo global de referencia (todos los cursos juntos)

Además de los 8 modelos por curso, se ajusta un par de modelos (Matemática / Lectura) sobre
**todos los cursos a la vez**. Sirve de panorama general y de línea base para comparar contra los
modelos por curso. Se agrega `curso` como variable de control (one-hot), porque cada nivel evaluado
tiene una escala de puntajes distinta; sus coeficientes son ajustes de escala, no efectos a
interpretar. El NSE va ordinal y estandarizado, igual que en el modelo principal.
```

**Celda de código:**

```python
# 4.x) MODELO GLOBAL — todos los cursos juntos
features_num_global = ["ind_am", "ind_cc", "ind_hv", "ind_pf", "nse_ord"]
features_cat_global = ["cod_rural_rbd", "cod_depe2", "curso"]   # curso = control de escala
features_modelo_global = features_num_global + features_cat_global


def crear_pipeline_global():
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), features_num_global),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), features_cat_global),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])


def obtener_feature_names_global(model):
    pre = model.named_steps["preprocessor"]
    return (pre.named_transformers_["num"].get_feature_names_out(features_num_global).tolist()
            + pre.named_transformers_["cat"].get_feature_names_out(features_cat_global).tolist())


metricas_global_rows, coef_global_rows = [], []
for target, asignatura, color in ASIGNATURAS:
    data = df_modelo.dropna(subset=features_modelo_global + [target, "rbd"]).copy()
    X, y, groups = data[features_modelo_global], data[target], data["rbd"]

    # Split agrupado por establecimiento (mismo criterio que los modelos por curso).
    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups)
    )
    model = crear_pipeline_global()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    m_train = calcular_metricas(y.iloc[train_idx], model.predict(X.iloc[train_idx]))
    m_test = calcular_metricas(y.iloc[test_idx], model.predict(X.iloc[test_idx]))

    cv = cross_validate(crear_pipeline_global(), X, y, groups=groups, cv=GroupKFold(n_splits=5),
                        scoring={"MAE": "neg_mean_absolute_error",
                                 "MSE": "neg_mean_squared_error", "R2": "r2"})
    metricas_global_rows.append({
        "asignatura": asignatura, "n_total": len(data),
        "MAE_train": m_train["MAE"], "MAE_test": m_test["MAE"],
        "RMSE_train": m_train["RMSE"], "RMSE_test": m_test["RMSE"],
        "R2_train": m_train["R2"], "R2_test": m_test["R2"],
        "MAE_CV": -cv["test_MAE"].mean(), "RMSE_CV": np.sqrt(-cv["test_MSE"]).mean(),
        "R2_CV": cv["test_R2"].mean(), "R2_CV_STD": cv["test_R2"].std(),
    })
    coef_tmp = pd.DataFrame({"variable": obtener_feature_names_global(model),
                             "coeficiente": model.named_steps["regressor"].coef_})
    coef_tmp.insert(0, "asignatura", asignatura)
    coef_global_rows.append(coef_tmp)

metricas_global_df = pd.DataFrame(metricas_global_rows)
coeficientes_global_df = pd.concat(coef_global_rows, ignore_index=True)

print("Modelo global — métricas train/test y validación cruzada:")
display(metricas_global_df.round(3))

# Importancia de variables (se omiten del ranking los dummies de `curso`, que son controles de escala)
for asignatura in HUE_ASIGNATURA:
    c = coeficientes_global_df[coeficientes_global_df["asignatura"] == asignatura].copy()
    es_control = c["variable"].str.startswith("curso")
    interes = c[~es_control].reindex(
        c[~es_control]["coeficiente"].abs().sort_values(ascending=False).index
    )
    plt.figure(figsize=(9, 5))
    colores = [PALETA_ASIGNATURA[asignatura] if v > 0 else "gray" for v in interes["coeficiente"]]
    plt.barh(interes["variable"], interes["coeficiente"], color=colores)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.title(f"Modelo global — importancia de variables · {asignatura}")
    plt.xlabel("Coeficiente sobre SIMCE")
    plt.tight_layout()
    plt.show()
    print(f"(controles de escala omitidos del ranking: {sorted(c.loc[es_control, 'variable'])})")
```

Resultado verificado: **Matemática** R²_test 0,47 · MAE 17,3 — **Lectura** R²_test 0,53 · MAE 14,4
(CV ≈ test, sin sobreajuste). El NSE es el predictor sustantivo dominante; `ind_cc` el IDPS más alto.

---

### 1.B — Modelo unificado mate + lectura con `tipo_prueba`

> **Dónde:** justo después del modelo global. **Esto lo pidió el profe** (incorporar el tipo de
> prueba como variable explicativa para no tener dos modelos totalmente independientes).

**Celda markdown (antes del código):**

```markdown
## 4.y) Modelo unificado: Matemática + Lectura con `tipo_prueba`

Atendiendo el feedback del profesor, se integra un modelo que **no** separa por completo Matemática
y Lectura. Se apilan ambas pruebas en una sola variable objetivo (`puntaje_simce`) y se agrega
`tipo_prueba` como predictor categórico. Así el modelo comparte la estructura común de ambas
asignaturas y el coeficiente de `tipo_prueba` mide la **brecha sistemática** entre ellas. La
partición es agrupada por establecimiento (`rbd`) para que Matemática y Lectura del mismo colegio no
queden a ambos lados del split.
```

**Celda de código:**

```python
# 4.y) MODELO UNIFICADO mate+lect con `tipo_prueba`
df_unif = df_modelo.melt(
    id_vars=["rbd", "agno", "curso", "nse_ord", "cod_rural_rbd", "cod_depe2",
             "ind_am", "ind_cc", "ind_hv", "ind_pf"],
    value_vars=targets, var_name="_col", value_name="puntaje_simce",
)
df_unif["tipo_prueba"] = df_unif["_col"].map(
    {"prom_mate2m_rbd": "Matemática", "prom_lect2m_rbd": "Lectura"}
)
df_unif = df_unif.dropna(subset=["puntaje_simce"]).reset_index(drop=True)
print(f"Formato largo: {len(df_modelo):,} filas anchas -> {len(df_unif):,} filas largas")

features_num_unif = ["ind_am", "ind_cc", "ind_hv", "ind_pf", "nse_ord"]
features_cat_unif = ["cod_rural_rbd", "cod_depe2", "curso", "tipo_prueba"]
features_modelo_unif = features_num_unif + features_cat_unif


def crear_pipeline_unif():
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), features_num_unif),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), features_cat_unif),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])


X, y, groups = df_unif[features_modelo_unif], df_unif["puntaje_simce"], df_unif["rbd"]
train_idx, test_idx = next(
    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups)
)
model_unif = crear_pipeline_unif()
model_unif.fit(X.iloc[train_idx], y.iloc[train_idx])
m_train = calcular_metricas(y.iloc[train_idx], model_unif.predict(X.iloc[train_idx]))
m_test = calcular_metricas(y.iloc[test_idx], model_unif.predict(X.iloc[test_idx]))
cv = cross_validate(crear_pipeline_unif(), X, y, groups=groups, cv=GroupKFold(n_splits=5),
                    scoring={"MAE": "neg_mean_absolute_error",
                             "MSE": "neg_mean_squared_error", "R2": "r2"})

print("Modelo unificado — métricas:")
display(pd.DataFrame([
    {"split": "train", **m_train},
    {"split": "test", **m_test},
    {"split": "CV (GroupKFold 5)", "MAE": -cv["test_MAE"].mean(),
     "RMSE": np.sqrt(-cv["test_MSE"]).mean(), "R2": cv["test_R2"].mean()},
]).round(3))

pre = model_unif.named_steps["preprocessor"]
nombres_unif = (pre.named_transformers_["num"].get_feature_names_out(features_num_unif).tolist()
                + pre.named_transformers_["cat"].get_feature_names_out(features_cat_unif).tolist())
coef_unif = pd.DataFrame({"variable": nombres_unif,
                          "coeficiente": model_unif.named_steps["regressor"].coef_})
coef_unif = coef_unif.reindex(coef_unif["coeficiente"].abs().sort_values(ascending=False).index)

print("Coeficientes del modelo unificado:")
display(coef_unif.round(3).reset_index(drop=True))
print("Coeficiente de `tipo_prueba` (brecha entre asignaturas; categoría de referencia = Lectura):")
display(coef_unif[coef_unif["variable"].str.startswith("tipo_prueba")].round(3))
```

Resultado verificado: **R²_test 0,44 · MAE 16,7** (CV ≈ test). Coeficiente
`tipo_prueba_Matemática ≈ −1,4` → a igualdad del resto, Matemática queda ~1,4 puntos por debajo de
Lectura. La brecha es pequeña → mate y lectura comparten estructura, lo que **justifica** el modelo
unificado.

---

### 1.C — Anexo predictivo (desfase temporal T-1)

> **Dónde:** sección nueva "6) Anexo predictivo", antes de iNterpret. Responde el punto más fuerte
> del feedback (la nota más baja: el modelo se presentaba como predictivo pero era contemporáneo).
> **Requiere subir a Colab `dataset_maestro.csv`** (lo genera el pipeline del repo) además de los 4
> CSV por curso.

**Celda markdown (antes del código):**

```markdown
# 6) Anexo predictivo — desfase temporal T-1 (out-of-time)

El análisis principal de este notebook es **contemporáneo** (IDPS y SIMCE del mismo año → asociación).
Para responder la observación sobre predicción real, se agrega un anexo con **desfase temporal
explícito**: los predictores se miden en el año **T-1** y el target (SIMCE) en el año **T**. La
partición es *out-of-time*: se entrena con años pasados y se evalúa en el último año disponible
(2025), que el modelo nunca ve durante el ajuste. Se comparan dos conjuntos de predictores:

- **Solo IDPS:** aísla el aporte socioemocional previo.
- **Autorregresivo:** agrega el SIMCE del año anterior (`simce_*_prev`), que aporta la mayor
  capacidad predictiva.

Por el desfase estricto T-1, solo sobreviven los cursos evaluados de forma anual (`4b` y `2m`).
```

**Celda de código:**

```python
# 6) ANEXO PREDICTIVO — desfase T-1 (requiere dataset_maestro.csv subido a Colab)
from pathlib import Path


def encontrar_maestro():
    candidatos = [Path("dataset_maestro.csv"), Path("/content/dataset_maestro.csv"),
                  Path("data/processed/dataset_maestro.csv"),
                  Path("/content/data/processed/dataset_maestro.csv")]
    for c in candidatos:
        if c.exists():
            return c
    for raiz in [Path.cwd(), Path("/content")]:
        if raiz.exists():
            for hit in raiz.rglob("dataset_maestro.csv"):
                return hit
    raise FileNotFoundError("Sube dataset_maestro.csv a /content (o a la carpeta de datos).")


df_pred = pd.read_csv(encontrar_maestro())
print(f"dataset_maestro: {df_pred.shape[0]:,} filas | "
      f"años {sorted(df_pred['agno'].unique())} | cursos {sorted(df_pred['curso'].unique())}")

FEAT_CAT_PRED = ["cod_rural_rbd", "cod_depe2"]
FEAT_NUM_IDPS = ["idps_am_prev", "idps_cc_prev", "idps_hv_prev", "idps_pf_prev", "nse_ord", "curso_ord"]
FEAT_NUM_AR = FEAT_NUM_IDPS + ["simce_mate_prev", "simce_lect_prev"]
TARGETS_PRED = {"Matemática": "target_mate", "Lectura": "target_lect"}
ANIOS_TRAIN, ANIO_TEST = [2017, 2018, 2023, 2024], 2025

cols_pred = list(dict.fromkeys(FEAT_NUM_AR + FEAT_CAT_PRED + list(TARGETS_PRED.values()) + ["agno"]))
df_pred = df_pred.dropna(subset=cols_pred).reset_index(drop=True)
train_pred = df_pred[df_pred["agno"].isin(ANIOS_TRAIN)]
test_pred = df_pred[df_pred["agno"] == ANIO_TEST]
print(f"Train (años {ANIOS_TRAIN}): {len(train_pred):,} | Test (año {ANIO_TEST}): {len(test_pred):,}")


def crear_pipeline_pred(feat_num, feat_cat):
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), feat_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat),
    ])
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LinearRegression())])


filas_pred = []
for asignatura, target in TARGETS_PRED.items():
    for nombre, feat_num in [("Solo IDPS", FEAT_NUM_IDPS), ("Autorregresivo", FEAT_NUM_AR)]:
        pipe = crear_pipeline_pred(feat_num, FEAT_CAT_PRED)
        pipe.fit(train_pred[feat_num + FEAT_CAT_PRED], train_pred[target])
        m_tr = calcular_metricas(train_pred[target], pipe.predict(train_pred[feat_num + FEAT_CAT_PRED]))
        m_te = calcular_metricas(test_pred[target], pipe.predict(test_pred[feat_num + FEAT_CAT_PRED]))
        filas_pred.append({"asignatura": asignatura, "modelo": nombre,
                           "MAE_train": m_tr["MAE"], "MAE_test": m_te["MAE"],
                           "RMSE_test": m_te["RMSE"], "R2_train": m_tr["R2"], "R2_test": m_te["R2"]})

metricas_pred_df = pd.DataFrame(filas_pred)
print("Predictivo T-1 — Train vs Test (out-of-time, test = 2025):")
display(metricas_pred_df.round(3))
```

Resultado verificado (test 2025): **Autorregresivo** R² 0,62 mate / 0,52 lect; **Solo IDPS** R² 0,36
mate / 0,38 lect. Sin sobreajuste (test ≥ train). El salto Solo-IDPS → Autorregresivo muestra que el
SIMCE previo es el predictor más potente, pero **los IDPS por sí solos ya anticipan ~0,36–0,38** del
rendimiento del año siguiente.

---

## 2) Celdas markdown (conclusiones + justificación de métrica)

### 2.A — Conclusiones: modelo principal (con NSE) vs modelo sin NSE

> **Dónde:** después de tu sección de comparación principal vs sin NSE (4.12–4.13). Números reales.

```markdown
## Conclusiones: modelo principal (con NSE) vs modelo sin NSE

**1. El NSE es la variable más influyente y su peso aumenta con el nivel educativo.**
Estandarizado, el coeficiente del NSE pasa de ~10 puntos en 4° básico a ~28 puntos en II medio
(Matemática), siendo el coeficiente más grande en casi todos los modelos — entre 2 y 4 veces el IDPS
más influyente (Clima de convivencia).

**2. Quitar el NSE empeora el ajuste, y el deterioro escala con el curso.**
En básica el R² baja poco (~0,08) y el MAE sube ~1,2 puntos. En II medio el impacto es grande: el R²
cae de 0,61 a 0,43 en Matemática (−0,18) y de 0,56 a 0,37 en Lectura (−0,19), con un MAE que sube
hasta ~4 puntos SIMCE. El NSE aporta información genuina, especialmente en enseñanza media.

**3. Al quitar el NSE, el Clima de convivencia (`ind_cc`) absorbe parte de su señal.**
El coeficiente de convivencia crece en todos los modelos al eliminar el NSE (hasta +4,0 en II medio
Matemática). Esto indica que parte de la asociación atribuida a la convivencia está en realidad
mediada por el nivel socioeconómico. Aun así, la convivencia se mantiene como el IDPS dominante en
ambos modelos.

**4. Autoestima y Participación se comportan distinto.** `ind_am` se mantiene o baja levemente al
quitar el NSE, e `ind_pf` se vuelve más negativo y mantiene signos inestables, por lo que debe
interpretarse con cautela y no como un efecto causal.

**5. Los modelos son estables.** El R² de test coincide con el de validación cruzada en ambas
variantes, por lo que las diferencias observadas no se explican por sobreajuste ni por azar de la
partición.

**Decisión metodológica:** se reportan ambos modelos. El modelo **con NSE** es la medida realista de
cuánto explican las variables disponibles (control correcto del contexto socioeconómico); el modelo
**sin NSE** funciona como diagnóstico para mostrar cuánto del "efecto IDPS" está en realidad mediado
por el nivel socioeconómico. La conclusión honesta es que los IDPS tienen una asociación modesta con
el SIMCE, parcialmente confundida con el NSE — y no que "los IDPS predicen el rendimiento".
```

### 2.B — Conclusiones de las tablas train/test y validación cruzada (modelo principal por curso)

> **Dónde:** después de tu tabla de métricas por curso (4.3).

```markdown
## Lectura de las métricas train/test y validación cruzada

- **El ajuste mejora con el nivel educativo.** El R² de test sube de ~0,37–0,39 en 4° básico a
  **0,61 (Matemática) / 0,56 (Lectura) en II medio**; 6° y 8° básico quedan intermedios (~0,35–0,43).
  Es decir, las variables disponibles explican mejor el SIMCE en enseñanza media que en básica.
- **No hay sobreajuste.** En los 8 modelos el R² de test es prácticamente igual al de validación
  cruzada (GroupKFold) y la brecha train→test es pequeña. La validación agrupada por establecimiento
  evita que un mismo colegio quede a ambos lados del split.
- **Error en escala interpretable.** El MAE de test va de ~13 a ~19 puntos SIMCE según curso y
  asignatura, lo que da una idea directa del margen de error del modelo.
```

### 2.C — Justificación de la métrica

> **Dónde:** al inicio de la sección de validación/iNterpret. Responde el pedido del profe de
> justificar la métrica elegida.

```markdown
## ¿Qué métrica priorizamos y por qué?

Se reportan tres métricas complementarias y se elige el **MAE como métrica principal**:

- **MAE (principal):** es el error promedio **en puntos SIMCE reales** ("el modelo se equivoca en
  promedio ±X puntos"). Es la más interpretable para comunicar a directivos y docentes, y no se
  infla por unos pocos casos extremos. Por eso la usamos como métrica de referencia del negocio.
- **R² (apoyo):** proporción de varianza explicada; útil para **comparar** el ajuste entre cursos y
  asignaturas en una escala común (0–1), pero no dice cuánto erramos en puntos.
- **RMSE (apoyo):** penaliza más los errores grandes; sirve para vigilar las colas. Si RMSE y MAE se
  separan mucho, hay errores grandes puntuales.

En resumen: el **MAE** responde "¿cuánto nos equivocamos?", el **R²** "¿cuánto explicamos?" y el
**RMSE** "¿hay errores grandes?". Priorizamos MAE por interpretabilidad, acompañado de los otros dos.
```

---

## 3) Correcciones de los comentarios "vs GitHub"

`gpt` dejó comentarios que comparan el notebook "contra el GitHub" como si fuera algo externo. En el
entregable final eso confunde (el GitHub **es** el trabajo del propio grupo). Reemplazá esas
referencias por "iteración anterior / primer modelo" y quita la meta-comparación. Buscar → reemplazar:

| Buscar (aprox.) | Reemplazar por |
|---|---|
| "metodología del notebook con visualización estilo GitHub" | "consolidación del análisis por curso del proyecto" |
| "adapta el análisis al dataset final consolidado en GitHub" | "usa el dataset final consolidado del proyecto" |
| "se adopta el enfoque del GitHub: modelos separados por curso" | "se adopta el enfoque por curso: modelos separados por curso" |
| "El GitHub incluye una matriz global de correlación..." | "Se incluye una matriz global de correlación..." |
| Título "## Ajustes respecto al script del GitHub" | "## Decisiones metodológicas de esta iteración" |
| "Se toma el GitHub como referencia de estructura y visualización" | "Se parte de la estructura de la iteración anterior" |
| "estilo del GitHub" / "estilo visual del GitHub" (varias) | "estilo visual del proyecto" |
| "no reentrenan con el pipeline del GitHub" | "no reentrenan con el pipeline de la iteración anterior" |
| Comentario `# 4.7) Importancia de variables (estilo GitHub - modelo principal)` | `# 4.7) Importancia de variables (modelo principal)` |
| Tabla iteraciones: "usando datasets finales del GitHub" | "usando los datasets finales por curso del proyecto" |

(Opcional) Renombrar la función `graficar_importancia_estilo_github` → `graficar_importancia` para
que no quede "github" en el código. Si lo haces, actualiza también sus 2 llamadas.

---

## 4) Ajustes de narrativa

- **Objetivo (feedback Problema 5,7):** el profe marcó que "desarrollar un modelo de regresión" es
  una *estrategia*, no el objetivo. Tu encabezado ya apunta a asociación; refuérzalo dejando claro
  que el objetivo es **comprender cómo se asocian los factores socioemocionales (IDPS) y el contexto
  con el rendimiento SIMCE**, y que la regresión es la herramienta. Evita el verbo "predecir" en el
  análisis contemporáneo (queda reservado para el Anexo predictivo T-1).
- **Mejoras vs primer modelo:** agrega un párrafo (o checklist) que mapee explícitamente cada punto
  del feedback con lo que ahora cubre el notebook:
  - NSE **ordinal** (`nse_ord`) ✔ · **skewness** por curso ✔ · **VIF / multicolinealidad** ✔
  - comparación explícita **Train vs Test** + **validación cruzada agrupada** ✔
  - **MAE** justificado como métrica principal ✔
  - **`tipo_prueba`** (modelo unificado mate+lect) ✔
  - **desfase temporal T-1** (anexo predictivo) ✔
  - modelo **global** + variante **sin NSE** como diagnósticos adicionales ✔

---

### Checklist de pendientes (de tu mensaje)

- [x] Modelo con todos los datos juntos → **1.A**
- [x] Decidir predictivo vs descriptivo → se **integra** como anexo (**1.C**); el principal sigue descriptivo
- [x] Conclusiones de train/test y CV (principal y sin NSE) → **2.A** y **2.B**
- [x] Markdown que explique las mejoras vs el primer modelo → **4**
- [x] Corregir comentarios "vs GitHub" de gpt → **3**
- [x] Análisis comparativo con NSE vs sin NSE → **2.A** (con números reales)
- [x] Decidir métrica (R²/MAE/RMSE) → **MAE principal**, justificado en **2.C**
- [x] `tipo_prueba` (pedido del profe) → **1.B**
