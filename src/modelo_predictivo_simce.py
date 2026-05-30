# %% [markdown]
# # Predicción anticipada del SIMCE a partir de indicadores socioemocionales (IDPS)
#
# **Enfoque predictivo temporal (out-of-time).** Se usan los indicadores de desarrollo
# personal y social (IDPS) y el SIMCE del año previo (**T-1**) para **anticipar** el SIMCE
# del año siguiente (**T**) a nivel de establecimiento.
#
# **Objetivo del proyecto:** anticipar y diagnosticar el desempeño académico (SIMCE) de los
# establecimientos a partir de variables contextuales y socioemocionales previas, para permitir
# la intervención temprana en recintos con riesgo escolar. (La regresión es la *herramienta*, no el fin.)
#
# Se entrenan **dos modelos independientes**: uno para Matemática (`target_mate`) y otro para
# Lectura (`target_lect`), porque los factores socioemocionales impactan distinto en cada prueba.
#
# Entrada: `data/processed/dataset_maestro.csv` (generado por `src/build_dataset_maestro.py`).
#
# Este archivo usa marcadores de celda `# %%`: VSCode / Jupytext lo abren y convierten a `.ipynb`.

# %% [markdown]
# ## Contrato de datos: `dataset_maestro.csv`
#
# Una fila por `(rbd, agno=T, curso)`.
#
# **Predictores medidos en T-1:** `idps_am_prev`, `idps_cc_prev`, `idps_hv_prev`, `idps_pf_prev`
# (IDPS 0-100) y `simce_mate_prev`, `simce_lect_prev` (SIMCE previo, componente autorregresivo).
#
# **Contexto (T-1):** `nse_ord` (ordinal 1=Bajo … 5=Alto), `curso_ord` (4b=4, 6b=6, 8b=8, 2m=10),
# `cod_rural_rbd`, `cod_depe2` (categóricas nominales).
#
# **Targets (año T):** `target_mate`, `target_lect`.
#
# > Nota: por el desfase estricto T-1 solo sobreviven los cursos evaluados de forma anual
# > (`4b` y `2m`); 6° y 8° básico se rinden en años alternados y no tienen pares consecutivos.

# %%
# Imports
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option("display.width", 120)
sns.set_theme(style="whitegrid")

# Intenta activar un backend interactivo (abre ventanas). Si no hay ninguno
# instalado, matplotlib se queda en 'Agg' y los gráficos solo se guardan como PNG.
for _bk in ("QtAgg", "TkAgg"):
    try:
        matplotlib.use(_bk, force=True)
        break
    except Exception:
        continue
_NO_INTERACTIVOS = ("agg", "pdf", "ps", "svg", "cairo", "template")
BACKEND_INTERACTIVO = matplotlib.get_backend().lower() not in _NO_INTERACTIVOS

# Carpeta donde se guardan los gráficos como PNG (versionados en git).
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def mostrar(nombre):
    """Guarda la figura como PNG (siempre) y abre ventana si hay backend interactivo.

    - En terminal sin entorno gráfico: solo guarda el PNG (sin warnings).
    - En notebook (VSCode/Jupyter): se ve inline gracias a plt.show().
    - En terminal con backend interactivo (Qt/Tk): además abre la ventana.
    """
    ruta = FIG_DIR / nombre
    plt.savefig(ruta, dpi=120, bbox_inches="tight")
    print(f"[figura guardada] {ruta}")
    if BACKEND_INTERACTIVO:
        plt.show()
    plt.close()

# %% [markdown]
# ## 1) OBTAIN — carga local del dataset maestro

# %%
def encontrar_dataset(nombre="dataset_maestro.csv"):
    """Busca data/processed/<nombre> subiendo desde el directorio actual."""
    for base in [Path.cwd(), *Path.cwd().parents]:
        ruta = base / "data" / "processed" / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError(f"No se encontro {nombre}. Ejecuta src/build_dataset_maestro.py primero.")


RUTA = encontrar_dataset()
df = pd.read_csv(RUTA)
print(f"Cargado: {RUTA}")
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
print(f"Anios-objetivo (T): {sorted(df['agno'].unique())}")
print(f"Cursos: {sorted(df['curso'].unique())}")
print(df.head())

# %% [markdown]
# ## 2) SCRUB — limpieza para modelado
#
# El maestro ya viene consolidado y con el desfase aplicado. Aquí solo se eliminan filas con
# nulos en las columnas del modelo (los IDPS previos tienen pocos faltantes) y se verifican tipos.
# No se imputa ni se escala todavía: el escalamiento vive en el `Pipeline` para evitar fuga.

# %%
COLS_MODELO = [
    "idps_am_prev", "idps_cc_prev", "idps_hv_prev", "idps_pf_prev",
    "simce_mate_prev", "simce_lect_prev", "nse_ord", "curso_ord",
    "cod_rural_rbd", "cod_depe2", "target_mate", "target_lect",
]

print("Nulos por columna (antes):")
print(df[COLS_MODELO].isnull().sum().to_string())

antes = len(df)
df = df.dropna(subset=COLS_MODELO).reset_index(drop=True)
print(f"\nFilas: {antes} -> {len(df)} (eliminadas {antes - len(df)} por nulos)")
print(f"\nFilas por anio-objetivo:\n{df['agno'].value_counts().sort_index().to_string()}")

# %% [markdown]
# ## 3) EXPLORE
#
# ### 3.1 Estadística descriptiva y simetría (skewness)
# La regresión lineal asume errores aproximadamente normales. Revisamos la asimetría de los
# predictores continuos y de los targets para decidir si conviene transformar (log / Box-Cox).

# %%
continuas = ["idps_am_prev", "idps_cc_prev", "idps_hv_prev", "idps_pf_prev",
             "simce_mate_prev", "simce_lect_prev", "target_mate", "target_lect"]
print(df[continuas].describe().round(2).T)

print("\nAsimetria (skewness):")
skew = df[continuas].apply(lambda s: stats.skew(s))
print(skew.round(3).to_string())
print("\n|skew| > 1 sugiere asimetria marcada; entre 0.5 y 1, moderada.")

# %%
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, col in zip(axes.flatten(), continuas):
    ax.hist(df[col], bins=40, color="steelblue", edgecolor="white")
    ax.set_title(f"{col}\nskew={stats.skew(df[col]):.2f}")
plt.suptitle("Distribucion de predictores continuos y targets", fontsize=14)
plt.tight_layout()
mostrar("01_histogramas.png")

# %% [markdown]
# ### 3.2 Multicolinealidad — Matriz de correlación de Pearson
# Se espera correlación alta entre los IDPS entre sí y entre `simce_mate_prev` y `simce_lect_prev`.

# %%
num_corr = ["idps_am_prev", "idps_cc_prev", "idps_hv_prev", "idps_pf_prev",
            "simce_mate_prev", "simce_lect_prev", "nse_ord", "curso_ord",
            "target_mate", "target_lect"]
corr = df[num_corr].corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Matriz de correlacion de Pearson")
plt.tight_layout()
mostrar("02_correlacion_pearson.png")

# %% [markdown]
# ### 3.3 Factor de Inflación de la Varianza (VIF)
# `VIF > 5` indica multicolinealidad problemática. Si aparece, justifica usar regularización
# (Ridge / ElasticNet) en lugar de OLS puro.

# %%
def tabla_vif(data, columnas):
    X = add_constant(data[columnas].astype(float))
    filas = []
    for i, col in enumerate(X.columns):
        filas.append({"variable": col, "VIF": variance_inflation_factor(X.values, i)})
    return pd.DataFrame(filas).query("variable != 'const'").sort_values("VIF", ascending=False)


feat_num_full = ["idps_am_prev", "idps_cc_prev", "idps_hv_prev", "idps_pf_prev",
                 "simce_mate_prev", "simce_lect_prev", "nse_ord", "curso_ord"]
vif = tabla_vif(df, feat_num_full)
print(vif.round(2).to_string(index=False))
print(f"\nVariables con VIF > 5: {vif.query('VIF > 5')['variable'].tolist()}")

# %% [markdown]
# ## 4) Feature engineering
#
# - **NSE** ya está como **Ordinal Encoding** (`nse_ord` 1..5) respetando la jerarquía socioeconómica.
# - **Curso** como ordinal (`curso_ord`) por nivel de escolaridad.
# - **Nominales** (`cod_rural_rbd`, `cod_depe2`) -> One-Hot dentro del pipeline.
# - Continuas -> `StandardScaler` dentro del pipeline.
#
# Definimos dos conjuntos de features para contrastar:
# - **Autorregresivo**: incluye `simce_*_prev` (máxima capacidad predictiva).
# - **Solo IDPS**: sin `simce_*_prev` (aísla el aporte socioemocional, narrativa del proyecto).

# %%
FEAT_CAT = ["cod_rural_rbd", "cod_depe2"]
FEAT_NUM_IDPS = ["idps_am_prev", "idps_cc_prev", "idps_hv_prev", "idps_pf_prev",
                 "nse_ord", "curso_ord"]
FEAT_NUM_AR = FEAT_NUM_IDPS + ["simce_mate_prev", "simce_lect_prev"]

TARGETS = {"Matematica": "target_mate", "Lectura": "target_lect"}

# %% [markdown]
# ## 5) MODEL — Partición temporal (Out-of-Time) y entrenamiento
#
# **Partición Out-of-Time**: el modelo se entrena con años pasados y se prueba en el **último año
# disponible (2025)**, que nunca se ve durante el ajuste. Esto evita la fuga de información que
# provocaría un `train_test_split` aleatorio sobre datos temporales.

# %%
ANIOS_TRAIN = [2017, 2018, 2023, 2024]
ANIO_TEST = 2025

train_df = df[df["agno"].isin(ANIOS_TRAIN)].sort_values("agno").reset_index(drop=True)
test_df = df[df["agno"] == ANIO_TEST].reset_index(drop=True)
print(f"Train (anios {ANIOS_TRAIN}): {len(train_df)} filas")
print(f"Test  (anio {ANIO_TEST}): {len(test_df)} filas")

# %%
def construir_pipeline(feat_num, feat_cat, estimador):
    pre = ColumnTransformer([
        ("num", StandardScaler(), feat_num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat),
    ])
    return Pipeline([("pre", pre), ("modelo", estimador)])


def metricas(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


resultados = []  # acumula metricas para la tabla comparativa final
modelos = {}     # guarda pipelines entrenados


def entrenar_evaluar(nombre_modelo, asignatura, target, feat_num, feat_cat, estimador):
    pipe = construir_pipeline(feat_num, feat_cat, estimador)
    Xtr, ytr = train_df[feat_num + feat_cat], train_df[target]
    Xte, yte = test_df[feat_num + feat_cat], test_df[target]
    pipe.fit(Xtr, ytr)
    m_tr = metricas(ytr, pipe.predict(Xtr))
    m_te = metricas(yte, pipe.predict(Xte))
    for split, m in [("train", m_tr), ("test", m_te)]:
        resultados.append({"modelo": nombre_modelo, "asignatura": asignatura,
                           "split": split, **m})
    modelos[(nombre_modelo, asignatura)] = pipe
    print(f"[{nombre_modelo} | {asignatura}]  "
          f"train MAE={m_tr['MAE']:.2f} R2={m_tr['R2']:.3f}  |  "
          f"test MAE={m_te['MAE']:.2f} RMSE={m_te['RMSE']:.2f} R2={m_te['R2']:.3f}")
    return pipe

# %% [markdown]
# ### 5.1 Regresión lineal (OLS) — modelo autorregresivo y modelo solo-IDPS

# %%
for asignatura, target in TARGETS.items():
    entrenar_evaluar("OLS autorregresivo", asignatura, target, FEAT_NUM_AR, FEAT_CAT,
                     LinearRegression())
    entrenar_evaluar("OLS solo-IDPS", asignatura, target, FEAT_NUM_IDPS, FEAT_CAT,
                     LinearRegression())

# %% [markdown]
# ### 5.2 Regularización (Ridge / ElasticNet)
# Dado el VIF alto, se entrena Ridge y ElasticNet con selección de hiperparámetros vía
# `TimeSeriesSplit` (validación cruzada que respeta el orden temporal).

# %%
tscv = TimeSeriesSplit(n_splits=4)
alphas = np.logspace(-3, 3, 25)

for asignatura, target in TARGETS.items():
    entrenar_evaluar("Ridge autorregresivo", asignatura, target, FEAT_NUM_AR, FEAT_CAT,
                     RidgeCV(alphas=alphas, cv=tscv))
    entrenar_evaluar("ElasticNet autorregresivo", asignatura, target, FEAT_NUM_AR, FEAT_CAT,
                     ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.9], cv=tscv, max_iter=5000))

# %% [markdown]
# ## 6) Reporte comparativo de métricas (Train vs Test)
#
# Justificación de la métrica principal: se prioriza el **MAE** por su interpretabilidad directa
# —está en **puntos SIMCE reales** ("el modelo se equivoca en promedio ±X puntos")—, lo que facilita
# la comunicación a directivos. Se acompaña de **RMSE** (penaliza errores grandes) y **R²**
# (proporción de varianza explicada). Una brecha grande train→test delataría sobreajuste.

# %%
tabla = pd.DataFrame(resultados)
pivote = tabla.pivot_table(index=["asignatura", "modelo"], columns="split",
                           values=["MAE", "RMSE", "R2"])
pivote = pivote.swaplevel(axis=1).sort_index(axis=1)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(pivote.to_string())

# %%
# Brecha de overfitting (test - train) en MAE y R2
piv = tabla.pivot_table(index=["asignatura", "modelo"], columns="split", values=["MAE", "R2"])
gap = pd.DataFrame({
    "MAE_train": piv[("MAE", "train")],
    "MAE_test": piv[("MAE", "test")],
    "MAE_gap": piv[("MAE", "test")] - piv[("MAE", "train")],
    "R2_train": piv[("R2", "train")],
    "R2_test": piv[("R2", "test")],
    "R2_gap": piv[("R2", "train")] - piv[("R2", "test")],
})
print(gap.round(3).to_string())
print("\nMAE_gap alto o R2_gap alto => posible sobreajuste.")

# %% [markdown]
# ## 7) Diagnósticos del modelo (iNterpret)
#
# Se diagnostica el mejor modelo por asignatura. Por defecto se usa el **Ridge autorregresivo**.

# %%
MODELO_DIAG = "Ridge autorregresivo"
FEAT_DIAG = FEAT_NUM_AR


def predichos_residuos(asignatura, target):
    pipe = modelos[(MODELO_DIAG, asignatura)]
    yte = test_df[target]
    pred = pipe.predict(test_df[FEAT_DIAG + FEAT_CAT])
    return yte.values, pred, yte.values - pred

# %% [markdown]
# ### 7.1 Gráfico de residuos (homocedasticidad) y normalidad

# %%
fig, axes = plt.subplots(len(TARGETS), 3, figsize=(18, 5 * len(TARGETS)))
for i, (asignatura, target) in enumerate(TARGETS.items()):
    y, pred, res = predichos_residuos(asignatura, target)
    ax = axes[i]
    ax[0].scatter(pred, res, alpha=0.3, s=12, color="steelblue")
    ax[0].axhline(0, color="red", ls="--")
    ax[0].set_title(f"{asignatura}: Residuos vs Predicho")
    ax[0].set_xlabel("Predicho")
    ax[0].set_ylabel("Residuo")
    ax[1].hist(res, bins=40, color="coral", edgecolor="white")
    ax[1].axvline(0, color="black", ls="--")
    ax[1].set_title(f"{asignatura}: Distribucion de residuos")
    stats.probplot(res, dist="norm", plot=ax[2])
    ax[2].set_title(f"{asignatura}: QQ-plot de residuos")
plt.tight_layout()
mostrar("03_residuos.png")

# %% [markdown]
# ### 7.2 Distancia de Cook — puntos influyentes
# Se ajusta un OLS (statsmodels) sobre el set de entrenamiento y se marcan los establecimientos
# cuya influencia supera el umbral D_i > 4 * mean(D).

# %%
def cook(asignatura, target, feat_num, feat_cat):
    X = pd.get_dummies(train_df[feat_num + feat_cat], columns=feat_cat, drop_first=True).astype(float)
    X = sm.add_constant(X)
    ols = sm.OLS(train_df[target].astype(float), X).fit()
    d = ols.get_influence().cooks_distance[0]
    umbral = 4 * d.mean()
    n_infl = int((d > umbral).sum())
    print(f"[{asignatura}] umbral=4*mean(D)={umbral:.5f}  ->  {n_infl} puntos influyentes "
          f"({100 * n_infl / len(d):.1f}%)")
    return d, umbral


fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 5))
influyentes = {}
for ax, (asignatura, target) in zip(axes, TARGETS.items()):
    d, umbral = cook(asignatura, target, FEAT_DIAG, FEAT_CAT)
    ax.scatter(np.arange(len(d)), d, s=6, color="steelblue", alpha=0.5)
    ax.axhline(umbral, color="red", ls="--", label="4*mean(D)")
    ax.set_title(f"{asignatura}: Distancia de Cook (train)")
    ax.set_xlabel("Indice")
    ax.set_ylabel("Cook's D")
    ax.legend()
    mask = d > umbral
    influyentes[asignatura] = train_df.loc[mask, ["rbd", "agno", "curso"]].assign(cook=d[mask])
plt.tight_layout()
mostrar("04_distancia_cook.png")

print("\nEjemplos de establecimientos influyentes (Matematica):")
print(influyentes["Matematica"].sort_values("cook", ascending=False).head(10).to_string(index=False))

# %% [markdown]
# ### 7.3 Importancia de variables (coeficientes del modelo de diagnóstico)

# %%
fig, axes = plt.subplots(1, len(TARGETS), figsize=(16, 6))
for ax, (asignatura, target) in zip(axes, TARGETS.items()):
    pipe = modelos[(MODELO_DIAG, asignatura)]
    pre = pipe.named_steps["pre"]
    nombres = (pre.named_transformers_["num"].get_feature_names_out(FEAT_DIAG).tolist()
               + pre.named_transformers_["cat"].get_feature_names_out(FEAT_CAT).tolist())
    coefs = pipe.named_steps["modelo"].coef_
    cdf = pd.DataFrame({"variable": nombres, "coef": coefs})
    cdf = cdf.reindex(cdf["coef"].abs().sort_values(ascending=False).index)
    colores = ["steelblue" if c > 0 else "coral" for c in cdf["coef"]]
    ax.barh(cdf["variable"], cdf["coef"], color=colores)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_title(f"{asignatura}: coeficientes ({MODELO_DIAG})")
plt.tight_layout()
mostrar("05_coeficientes.png")

# %% [markdown]
# ## Conclusiones (a completar tras ejecutar)
#
# - **Predictivo, no contemporáneo:** las features son del año T-1 y el target del año T.
# - **Dos modelos** (Matemática y Lectura) con desempeño y coeficientes propios.
# - **NSE ordinal**, **VIF/correlación** evaluados, **regularización** aplicada por la multicolinealidad.
# - **Partición Out-of-Time** (test = 2025) y **comparación Train vs Test** para descartar overfitting.
# - **MAE** como métrica de negocio (puntos SIMCE), con RMSE y R² de apoyo.
# - **Residuos** (homocedasticidad/normalidad) y **Distancia de Cook** (puntos influyentes) revisados.
# - El modelo **solo-IDPS** aísla el aporte socioemocional; el **autorregresivo** maximiza la predicción.
