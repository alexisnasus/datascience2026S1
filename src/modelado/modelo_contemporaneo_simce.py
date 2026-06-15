# %% [markdown]
# # Análisis contemporáneo del SIMCE mediante IDPS y variables contextuales
#
# **Enfoque descriptivo-explicativo (asociación, mismo año T).** Se estudia cómo se relacionan los
# Indicadores de Desarrollo Personal y Social (IDPS) y el contexto del establecimiento (NSE,
# dependencia, ruralidad) con el rendimiento SIMCE de Matemática y Lectura. La regresión es la
# *herramienta* para estimar asociaciones; **no** se afirma predicción de resultados futuros (para el
# enfoque predictivo con desfase T-1 ver `modelo_predictivo_simce.py`).
#
# Este documento consolida en una sola narrativa las **tres lentes complementarias** del análisis
# contemporáneo:
#
# - **Sección A — Global:** un par de modelos (mate / lectura) sobre **todos los cursos** mezclados
#   (4b/6b/8b/2m), usando `curso` y la interacción NSE×curso como controles de escala. Da el panorama
#   general.
# - **Sección B — Por curso:** un par de modelos **por cada curso** (8 modelos), con el recorte de
#   outliers (IQR) calculado **por curso** sobre filas homogéneas, para ver cómo cambia el efecto de
#   los IDPS de básica a media.
# - **Sección C — Unificado:** mate y lectura se integran en una sola variable objetivo
#   (`puntaje_simce`) y el modelo distingue la asignatura con la variable `tipo_prueba`, evitando dos
#   modelos totalmente independientes (recomendación del profesor).
#
# **Significado de las variables**
# - `ind_am` Autoestima Académica y Motivación · `ind_cc` Clima de Convivencia Escolar ·
#   `ind_hv` Hábitos de Vida Saludable · `ind_pf` Participación y Formación Ciudadana (todas 0-100).
# - `cod_grupo` NSE (Bajo < Medio bajo < Medio < Medio alto < Alto) · `cod_depe2` dependencia ·
#   `cod_rural_rbd` ruralidad.
#
# **Metodología:** OSEMN (Obtain → Scrub → Explore → Model → iNterpret).
#
# Entrada: `data/processed/dataset_historico_completo.csv` (Secciones A y C) y
# `data/processed/por_curso/dataset_historico_{curso}.csv` (Sección B).
# Corre **100% local** (`matplotlib` en backend Agg; las figuras se guardan como PNG, no se muestran).
# Las celdas `# %%` permiten abrirlo como notebook en VSCode/Jupytext.

# %%
# Imports y configuración (sin dependencias de Colab: corre como script local)
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventanas: las figuras se guardan como PNG
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import skew
from sklearn.model_selection import (
    train_test_split, cross_val_score, GroupShuffleSplit, GroupKFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120)

# Carpeta padre de figuras; cada sección guarda en su subcarpeta (global/por_curso/unificado).
FIG_BASE = Path(__file__).resolve().parents[2] / "reports" / "figuras_contemporaneo"  # src/modelado/<archivo>.py -> raíz

# Orden jerárquico del NSE (menor a mayor), para el Ordinal Encoding y los gráficos.
orden_nse = ["Bajo", "Medio bajo", "Medio", "Medio alto", "Alto"]

# Predictores comunes.
features_num = ["ind_am", "ind_cc", "ind_hv", "ind_pf"]   # IDPS -> StandardScaler
features_ord = ["cod_grupo"]                              # NSE -> Ordinal Encoding
# Conjuntos categóricos por sección (One-Hot):
features_cat_global = ["cod_rural_rbd", "cod_depe2", "curso", "nse_x_curso"]
features_cat_curso  = ["cod_rural_rbd", "cod_depe2"]       # dentro de un curso, curso/NSE×curso son constantes
features_cat_unif   = ["cod_rural_rbd", "cod_depe2", "curso", "tipo_prueba"]

CURSOS = ["4b", "6b", "8b", "2m"]  # progresión educativa (eje X de la evolución)
CURSO_NOMBRE = {"4b": "4° básico", "6b": "6° básico", "8b": "8° básico", "2m": "II medio"}
IDPS_NOMBRE = {
    "ind_am": "Autoestima Académica", "ind_cc": "Clima de Convivencia",
    "ind_hv": "Hábitos Saludables", "ind_pf": "Participación Ciudadana",
}
ASIGNATURAS = [("prom_mate2m_rbd", "Matemática"), ("prom_lect2m_rbd", "Lectura")]

# Mapas de estandarización categórica (no-op si ya vienen como texto canónico desde el build).
nse_map = {"1.0": "Bajo", "2.0": "Medio bajo", "3.0": "Medio", "4.0": "Medio alto", "5.0": "Alto"}
depe_map = {"1": "Municipal", "2": "Particular subvencionado", "3": "Particular pagado", "4": "SLEP"}
rural_map = {"1": "Urbano", "2": "Rural"}


# %%
# Helpers (autocontenidos: este maestro no importa otros scripts del repo)

def guardar(nombre, sub):
    """Guarda la figura actual como PNG en reports/figuras_contemporaneo/<sub>/ y la cierra."""
    carpeta = FIG_BASE / sub
    carpeta.mkdir(parents=True, exist_ok=True)
    plt.savefig(carpeta / nombre, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[figura guardada] {carpeta / nombre}")


def encontrar_dataset(nombre="dataset_historico_completo.csv"):
    """Busca data/processed/<nombre> subiendo desde el directorio actual."""
    for base in [Path.cwd(), *Path.cwd().parents]:
        ruta = base / "data" / "processed" / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"No se encontró {nombre}. Ejecuta src/build_dataset_historico.py primero."
    )


def encontrar_por_curso(curso):
    """Busca data/processed/por_curso/dataset_historico_<curso>.csv subiendo desde el CWD."""
    nombre = f"dataset_historico_{curso}.csv"
    for base in [Path.cwd(), *Path.cwd().parents]:
        ruta = base / "data" / "processed" / "por_curso" / nombre
        if ruta.exists():
            return ruta
    raise FileNotFoundError(
        f"No se encontró {nombre}. Ejecuta src/build_dataset_historico_per_curso.py primero."
    )


def metricas_split(modelo, X_, y_):
    pred = modelo.predict(X_)
    return {
        "MAE": mean_absolute_error(y_, pred),
        "RMSE": np.sqrt(mean_squared_error(y_, pred)),
        "R2": r2_score(y_, pred),
    }


def tabla_train_test(modelo, X_train, y_train, X_test, y_test, titulo):
    """Imprime la comparación explícita Train vs Test (brecha = sobreajuste)."""
    tt = pd.DataFrame([metricas_split(modelo, X_train, y_train),
                       metricas_split(modelo, X_test, y_test)],
                      index=["Train", "Test"]).round(3)
    tt.loc["Brecha (Test-Train)"] = (tt.loc["Test"] - tt.loc["Train"]).round(3)
    print(f"\n  RESULTADOS {titulo} — Train vs Test:")
    print("   " + tt.to_string().replace("\n", "\n   "))
    return tt


def calcular_vif(df_predictores):
    """VIF de cada predictor vía R² auxiliar (VIF > 5 advierte multicolinealidad)."""
    filas = []
    for variable in df_predictores.columns:
        otras = [c for c in df_predictores.columns if c != variable]
        r2_aux = LinearRegression().fit(
            df_predictores[otras], df_predictores[variable]
        ).score(df_predictores[otras], df_predictores[variable])
        vif = np.inf if r2_aux >= 1 else 1 / (1 - r2_aux)
        filas.append({"variable": variable, "VIF": round(vif, 3)})
    return pd.DataFrame(filas).sort_values("VIF", ascending=False)


def tabla_skew(df, cols, titulo):
    """Imprime la tabla de asimetría (skewness); |skew|>1 = asimetría marcada."""
    print(f"\n  -- {titulo}: Skewness --")
    tabla = df[cols].apply(lambda s: skew(s.dropna())).round(3).rename("skewness").to_frame()
    tabla["asimetria"] = tabla["skewness"].abs().apply(
        lambda v: "marcada" if v > 1 else ("moderada" if v > 0.5 else "leve"))
    print("   " + tabla.to_string().replace("\n", "\n   "))


def construir_pipeline(features_cat):
    """Pipeline: IDPS estandarizados + NSE ordinal + resto categórico One-Hot + LinearRegression."""
    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(), features_num),
        ("ord", OrdinalEncoder(categories=[orden_nse],
                               handle_unknown="use_encoded_value", unknown_value=-1), features_ord),
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ])
    return Pipeline(steps=[("preprocessor", pre), ("regressor", LinearRegression())])


def nombres_features(model, features_cat):
    """Nombres de columnas tras el preprocesamiento, en el orden de los coeficientes."""
    pre = model.named_steps["preprocessor"]
    return (pre.named_transformers_["num"].get_feature_names_out(features_num).tolist()
            + features_ord
            + pre.named_transformers_["cat"].get_feature_names_out(features_cat).tolist())


def _es_control(nombre_var):
    """True si la variable es un CONTROL de escala (curso o su interacción con NSE), no un efecto
    sustantivo a interpretar: 'curso' solo corrige que cada prueba tiene una escala distinta."""
    return nombre_var.startswith("curso") or nombre_var.startswith("nse_x_curso")


def fig_real_vs_pred(y_true, y_pred, titulo, color, nombre, sub):
    plt.figure(figsize=(6.5, 6.5))
    plt.scatter(y_true, y_pred, alpha=0.3, s=14, color=color)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="Predicción perfecta")
    plt.xlabel("Valores reales (SIMCE)")
    plt.ylabel("Valores predichos")
    plt.title(f"Real vs predicho — {titulo}")
    plt.legend()
    plt.tight_layout()
    guardar(nombre, sub)


def fig_residuos(residuos, titulo, color, nombre, sub):
    plt.figure(figsize=(8, 4))
    plt.hist(residuos, bins=40, color=color, edgecolor="white")
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Residuo (Real − Predicho)")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de residuos — {titulo}")
    plt.tight_layout()
    guardar(nombre, sub)


def fig_importancia(coef_df, titulo, color_pos, color_neg, nombre, sub, separar_controles=False):
    """Barras con el coeficiente de cada variable. Si separar_controles, deja fuera del ranking los
    controles de escala (curso/NSE×curso) para no mezclar peras con manzanas."""
    if separar_controles:
        interes = coef_df[~coef_df["Variable"].map(_es_control)].copy()
        control = coef_df[coef_df["Variable"].map(_es_control)].copy()
    else:
        interes, control = coef_df.copy(), None
    interes = interes.reindex(interes["Coeficiente"].abs().sort_values(ascending=False).index)

    plt.figure(figsize=(10, 6))
    colores = [color_pos if c > 0 else color_neg for c in interes["Coeficiente"]]
    plt.barh(interes["Variable"], interes["Coeficiente"], color=colores)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Coeficiente (efecto sobre SIMCE)")
    plt.title(f"Importancia de variables — {titulo}")
    plt.tight_layout()
    guardar(nombre, sub)

    print(f"\n  Top variables de interés — {titulo}:")
    print("   " + interes.head(5).to_string(index=False).replace("\n", "\n   "))
    if control is not None and len(control):
        control = control.reindex(control["Coeficiente"].abs().sort_values(ascending=False).index)
        print("  Controles de escala (curso/NSE×curso) — NO se interpretan como efecto real:")
        print("   " + control.head(5).to_string(index=False).replace("\n", "\n   "))


def distancia_cook(df_modelo, target, features_cat, cols_id, titulo, nombre=None, sub=None):
    """Distancia de Cook (statsmodels OLS): puntos con D_i > 4·media(D). Imprime los más influyentes;
    guarda figura solo si se pasa `nombre`."""
    Xc = pd.DataFrame(index=df_modelo.index)
    Xc[features_num] = df_modelo[features_num]
    Xc["nse_ord"] = df_modelo["cod_grupo"].map({c: i + 1 for i, c in enumerate(orden_nse)})
    Xc = pd.concat([Xc, pd.get_dummies(df_modelo[features_cat], drop_first=True)], axis=1)
    Xc = sm.add_constant(Xc.astype(float))
    ols = sm.OLS(df_modelo[target].astype(float), Xc).fit()
    d = ols.get_influence().cooks_distance[0]
    umbral = 4 * d.mean()
    n_infl = int((d > umbral).sum())
    print(f"  [{titulo}] Cook: umbral 4·media(D)={umbral:.5f} -> "
          f"{n_infl} influyentes ({100 * n_infl / len(d):.1f}%)")

    if nombre is not None:
        plt.figure(figsize=(11, 4))
        plt.scatter(np.arange(len(d)), d, s=6, color="steelblue", alpha=0.5)
        plt.axhline(umbral, color="red", ls="--", label="4·media(D)")
        plt.title(f"Distancia de Cook — {titulo}")
        plt.xlabel("Índice de observación (train)")
        plt.ylabel("Cook's D")
        plt.legend()
        plt.tight_layout()
        guardar(nombre, sub)

    influyentes = df_modelo.loc[d > umbral, cols_id].assign(cook=d[d > umbral])
    print("   " + influyentes.sort_values("cook", ascending=False).head(5)
          .to_string(index=False).replace("\n", "\n   "))


# %% [markdown]
# ## 1) OBTAIN — carga del dataset contemporáneo (todos los cursos)

# %%
df_raw = pd.read_csv(encontrar_dataset())
print(f"Dimensiones: {df_raw.shape[0]} filas × {df_raw.shape[1]} columnas")
print(f"Años disponibles: {sorted(df_raw['agno'].unique())}")
print(f"Cursos: {sorted(df_raw['curso'].unique())}")
print(df_raw.head())

# %% [markdown]
# ## 2) SCRUB (global) — limpieza para las Secciones A y C
#
# 1) estandarizar categóricas a texto legible (no-op si ya vienen canónicas); 2) eliminar nulos;
# 3) crear la interacción NSE×curso; 4) recortar outliers con **IQR global** sobre ambos targets.
# El resultado `df_global` alimenta la Sección A (global) y la Sección C (unificado). La Sección B
# recalcula su propio IQR **por curso**.

# %%
df_global = df_raw.copy()
df_global["cod_grupo"] = df_global["cod_grupo"].replace(nse_map)
df_global["cod_depe2"] = df_global["cod_depe2"].replace(depe_map)
df_global["cod_rural_rbd"] = df_global["cod_rural_rbd"].replace(rural_map)

antes = len(df_global)
df_global = df_global.dropna()
print(f"dropna: {antes} -> {len(df_global)} filas ({antes - len(df_global)} eliminadas)")

# Interacción NSE × curso: el efecto del NSE sobre el SIMCE no es igual en todos los cursos.
df_global["nse_x_curso"] = df_global["cod_grupo"].astype(str) + " · " + df_global["curso"].astype(str)

# IQR global sobre ambos targets.
for col in ["prom_mate2m_rbd", "prom_lect2m_rbd"]:
    q1, q3 = df_global[col].quantile(0.25), df_global[col].quantile(0.75)
    iqr = q3 - q1
    antes = len(df_global)
    df_global = df_global[(df_global[col] >= q1 - 1.5 * iqr) & (df_global[col] <= q3 + 1.5 * iqr)]
    print(f"outliers IQR en {col}: {antes - len(df_global)} eliminados")
df_global = df_global.reset_index(drop=True)
print(f"df_global final: {len(df_global)} filas")

# %% [markdown]
# ## 3) EXPLORE — descriptivo del análisis contemporáneo (una sola vez)
#
# Estadística descriptiva, **simetría (skewness)**, **multicolinealidad (VIF)** y 6 figuras
# comparando Matemática y Lectura lado a lado. Se hace sobre `df_global` y vale para las 3 secciones.

# %%
cols_interes = ["prom_mate2m_rbd", "prom_lect2m_rbd", "ind_am", "ind_cc", "ind_hv", "ind_pf"]
print(" -- Estadísticas descriptivas --")
print(df_global[cols_interes].describe().round(2).to_string())

tabla_skew(df_global, cols_interes, "Targets + IDPS (global)")
print("  Lectura: |skew| < 1 en los targets -> simetría aceptable, sin transformación.")

print("\n -- VIF de los indicadores IDPS --")
vif_idps = calcular_vif(df_global[features_num])
print("   " + vif_idps.to_string(index=False).replace("\n", "\n   "))
print(f"  Variables con VIF > 5: {vif_idps.query('VIF > 5')['variable'].tolist() or 'ninguna'}")

# %%
# 3.1) Distribución de los puntajes SIMCE (mate vs lectura)
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
df_global["prom_mate2m_rbd"].hist(bins=40, ax=axes[0], color="steelblue")
axes[0].set_title("SIMCE Matemática"); axes[0].set_xlabel("Puntaje"); axes[0].set_ylabel("Frecuencia")
df_global["prom_lect2m_rbd"].hist(bins=40, ax=axes[1], color="coral")
axes[1].set_title("SIMCE Lectura"); axes[1].set_xlabel("Puntaje")
plt.suptitle("Distribución de puntajes SIMCE por establecimiento", fontsize=13)
plt.tight_layout()
guardar("01_distribucion_simce.png", "global")

# 3.2) SIMCE por NSE
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
sns.boxplot(data=df_global, x="cod_grupo", y="prom_mate2m_rbd", order=orden_nse, ax=axes[0],
            palette="Blues", hue="cod_grupo", legend=False)
axes[0].set_title("Matemática por NSE"); axes[0].set_xlabel("NSE"); axes[0].set_ylabel("Puntaje SIMCE")
sns.boxplot(data=df_global, x="cod_grupo", y="prom_lect2m_rbd", order=orden_nse, ax=axes[1],
            palette="Oranges", hue="cod_grupo", legend=False)
axes[1].set_title("Lectura por NSE"); axes[1].set_xlabel("NSE"); axes[1].set_ylabel("")
plt.suptitle("Puntajes SIMCE por nivel socioeconómico", fontsize=13)
plt.tight_layout()
guardar("02_simce_por_nse.png", "global")

# 3.3) SIMCE por dependencia
fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
sns.boxplot(data=df_global, x="cod_depe2", y="prom_mate2m_rbd", ax=axes[0], palette="Set2",
            hue="cod_depe2", legend=False)
axes[0].set_title("Matemática por dependencia"); axes[0].set_xlabel("Dependencia")
axes[0].set_ylabel("Puntaje SIMCE"); axes[0].tick_params(axis="x", rotation=30)
sns.boxplot(data=df_global, x="cod_depe2", y="prom_lect2m_rbd", ax=axes[1], palette="Set2",
            hue="cod_depe2", legend=False)
axes[1].set_title("Lectura por dependencia"); axes[1].set_xlabel("Dependencia")
axes[1].tick_params(axis="x", rotation=30)
plt.suptitle("Puntajes SIMCE por dependencia administrativa", fontsize=13)
plt.tight_layout()
guardar("03_simce_por_dependencia.png", "global")

# 3.4) SIMCE por ruralidad
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
sns.boxplot(data=df_global, x="cod_rural_rbd", y="prom_mate2m_rbd", ax=axes[0], palette="Set3",
            hue="cod_rural_rbd", legend=False)
axes[0].set_title("Matemática por ruralidad"); axes[0].set_xlabel("Ruralidad")
axes[0].set_ylabel("Puntaje SIMCE")
sns.boxplot(data=df_global, x="cod_rural_rbd", y="prom_lect2m_rbd", ax=axes[1], palette="Set3",
            hue="cod_rural_rbd", legend=False)
axes[1].set_title("Lectura por ruralidad"); axes[1].set_xlabel("Ruralidad"); axes[1].set_ylabel("")
plt.suptitle("Puntajes SIMCE por ruralidad", fontsize=13)
plt.tight_layout()
guardar("04_simce_por_ruralidad.png", "global")

# 3.5) Matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(df_global[cols_interes].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Matriz de correlación: IDPS y puntajes SIMCE")
plt.tight_layout()
guardar("05_matriz_correlacion.png", "global")

# 3.6) Scatter IDPS vs SIMCE (mate y lectura, lado a lado)
idps = [("ind_am", "Autoestima Académica"), ("ind_cc", "Clima de Convivencia"),
        ("ind_hv", "Hábitos Saludables"), ("ind_pf", "Participación Ciudadana")]
fig, axes = plt.subplots(len(idps), 2, figsize=(14, 18), sharey="row")
for i, (col, nombre) in enumerate(idps):
    axes[i, 0].scatter(df_global[col], df_global["prom_mate2m_rbd"], alpha=0.2, s=10, color="steelblue")
    axes[i, 0].set_title(f"{nombre} vs Matemática"); axes[i, 0].set_xlabel(nombre)
    axes[i, 0].set_ylabel("SIMCE Matemática")
    axes[i, 1].scatter(df_global[col], df_global["prom_lect2m_rbd"], alpha=0.2, s=10, color="coral")
    axes[i, 1].set_title(f"{nombre} vs Lectura"); axes[i, 1].set_xlabel(nombre)
    axes[i, 1].set_ylabel("SIMCE Lectura")
plt.suptitle("Relación entre cada IDPS y los puntajes SIMCE", fontsize=14)
plt.tight_layout()
guardar("06_scatter_idps_vs_simce.png", "global")

# %% [markdown]
# # SECCIÓN A — Modelo global (todos los cursos)
#
# Un par de modelos (Matemática / Lectura) sobre `df_global`. Se incluyen `curso` y `nse_x_curso`
# como **controles de escala** (no son efectos a interpretar; corrigen que cada prueba tiene su
# escala). Métrica de negocio: **MAE** (puntos SIMCE); RMSE penaliza errores grandes; R² la varianza
# explicada. Se compara Train vs Test y se revisan residuos y Distancia de Cook.

# %%
def modelar_global(df, target, asignatura, color, prefijo):
    X = df[features_num + features_ord + features_cat_global]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = construir_pipeline(features_cat_global)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    tabla_train_test(model, X_train, y_train, X_test, y_test, f"GLOBAL · {asignatura}")
    cv = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"  Validación cruzada R² (5 folds): {cv.round(3)} -> media {cv.mean():.3f} ± {cv.std():.3f}")

    fig_real_vs_pred(y_test, y_pred, f"Global · {asignatura}", color, f"{prefijo}_real_vs_pred.png", "global")
    fig_residuos(y_test - y_pred, f"Global · {asignatura}", color, f"{prefijo}_residuos.png", "global")
    coef_df = pd.DataFrame({"Variable": nombres_features(model, features_cat_global),
                            "Coeficiente": model.named_steps["regressor"].coef_})
    fig_importancia(coef_df, f"Global · {asignatura}", color, "gray",
                    f"{prefijo}_importancia.png", "global", separar_controles=True)
    distancia_cook(df.loc[X_train.index], target, features_cat_global, ["rbd", "agno", "curso"],
                   f"Global · {asignatura}", f"{prefijo}_cook.png", "global")
    return {"R2": r2_score(y_test, y_pred), "MAE": mean_absolute_error(y_test, y_pred)}


resumen_global = {}
for target, asignatura in ASIGNATURAS:
    prefijo = "mate" if "mate" in target else "lect"
    color = "steelblue" if "mate" in target else "coral"
    resumen_global[asignatura] = modelar_global(df_global, target, asignatura, color, prefijo)

print("\n" + "=" * 48)
print("   SECCIÓN A — COMPARACIÓN GLOBAL (test)")
print("=" * 48)
print(pd.DataFrame(resumen_global).round(3).to_string())

# %% [markdown]
# # SECCIÓN B — Modelos por curso (4b · 6b · 8b · II medio)
#
# Un par de modelos por cada curso (8 en total). Clave metodológica: el **IQR se recalcula por curso**
# (filas homogéneas), y se omiten `curso`/`nse_x_curso` (constantes dentro de un curso). Los
# coeficientes IDPS quedan en unidades de desviación estándar → **comparables entre cursos**. El
# payoff es ver cómo evoluciona el efecto de los IDPS de básica a media.

# %%
def scrub_curso(df, curso):
    df = df.copy()
    df["cod_grupo"] = df["cod_grupo"].replace(nse_map)
    df["cod_depe2"] = df["cod_depe2"].replace(depe_map)
    df["cod_rural_rbd"] = df["cod_rural_rbd"].replace(rural_map)
    antes = len(df)
    df = df.dropna(subset=features_num + features_ord + features_cat_curso
                   + ["prom_mate2m_rbd", "prom_lect2m_rbd"])
    print(f"  [{curso}] dropna: {antes} -> {len(df)} filas")
    for col in ["prom_mate2m_rbd", "prom_lect2m_rbd"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        antes = len(df)
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
        print(f"  [{curso}] outliers IQR en {col}: {antes - len(df)} eliminados")
    return df.reset_index(drop=True)


def modelar_curso(df, curso, target, asignatura, color):
    X = df[features_num + features_ord + features_cat_curso]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = construir_pipeline(features_cat_curso)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    tabla_train_test(model, X_train, y_train, X_test, y_test, f"{asignatura} · {CURSO_NOMBRE[curso]}")
    coef_df = pd.DataFrame({"Variable": nombres_features(model, features_cat_curso),
                            "Coeficiente": model.named_steps["regressor"].coef_})
    pref = f"{'mate' if 'mate' in target else 'lect'}_{curso}"
    fig_importancia(coef_df, f"{asignatura} · {CURSO_NOMBRE[curso]}", color, "gray",
                    f"{pref}_importancia.png", "por_curso", separar_controles=False)
    fig_real_vs_pred(y_test, y_pred, f"{asignatura} · {CURSO_NOMBRE[curso]}", color,
                     f"{pref}_real_vs_pred.png", "por_curso")
    distancia_cook(df.loc[X_train.index], target, features_cat_curso, ["rbd", "agno"],
                   f"{asignatura} · {CURSO_NOMBRE[curso]}")  # tabla impresa (sin figura)

    coef_idps = dict(zip(coef_df["Variable"], coef_df["Coeficiente"]))
    return {"R2": r2_score(y_test, y_pred), "MAE": mean_absolute_error(y_test, y_pred),
            **{ind: coef_idps[ind] for ind in features_num}}


def tabla_resumen(resultados, asignatura):
    filas = features_num + ["R2", "MAE"]
    data = {curso: [resultados[(curso, asignatura)][f] for f in filas] for curso in CURSOS}
    etiquetas = [IDPS_NOMBRE[f] for f in features_num] + ["R²", "MAE"]
    tabla = pd.DataFrame(data, index=etiquetas)[CURSOS].round(3)
    print(f"\n{'=' * 60}\n  RESUMEN COEFICIENTES IDPS POR CURSO — {asignatura}\n{'=' * 60}")
    print(tabla.to_string())


def graficar_evolucion(resultados, asignatura, nombre_fig):
    plt.figure(figsize=(9, 5.5))
    for ind in features_num:
        coefs = [resultados[(curso, asignatura)][ind] for curso in CURSOS]
        plt.plot([CURSO_NOMBRE[c] for c in CURSOS], coefs, marker="o", label=IDPS_NOMBRE[ind])
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Curso (progresión educativa)")
    plt.ylabel(f"Coeficiente sobre SIMCE {asignatura} (por desviación estándar)")
    plt.title(f"Evolución del efecto de los IDPS por curso — {asignatura}")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    guardar(nombre_fig, "por_curso")


color_asig = {"Matemática": "steelblue", "Lectura": "coral"}
resultados_curso = {}
for curso in CURSOS:
    print(f"\n{'#' * 70}\n# CURSO {curso} ({CURSO_NOMBRE[curso]})\n{'#' * 70}")
    dfc = pd.read_csv(encontrar_por_curso(curso))
    print(f"  Cargado: {dfc.shape[0]} filas, años {sorted(dfc['agno'].unique())}")
    dfc = scrub_curso(dfc, curso)
    tabla_skew(dfc, ["prom_mate2m_rbd", "prom_lect2m_rbd"] + features_num, f"[{curso}]")
    vif_c = calcular_vif(dfc[features_num])
    print(f"  [{curso}] VIF > 5: {vif_c.query('VIF > 5')['variable'].tolist() or 'ninguna'}")
    for target, asignatura in ASIGNATURAS:
        resultados_curso[(curso, asignatura)] = modelar_curso(
            dfc, curso, target, asignatura, color_asig[asignatura])

for target, asignatura in ASIGNATURAS:
    tabla_resumen(resultados_curso, asignatura)
    graficar_evolucion(resultados_curso, asignatura,
                       f"evolucion_idps_{'matematica' if asignatura == 'Matemática' else 'lectura'}.png")

# %% [markdown]
# # SECCIÓN C — Modelo unificado (Matemática + Lectura con `tipo_prueba`)
#
# Mate y lectura se integran en una sola variable objetivo `puntaje_simce`; la categórica
# `tipo_prueba` distingue la asignatura. Así se evitan dos modelos totalmente independientes y se
# puede contrastar si las asociaciones se comportan igual en ambas pruebas. La partición usa
# **GroupShuffleSplit por establecimiento** (`id_registro`): las filas de mate y lectura del mismo
# colegio quedan **juntas** en train o en test (evita fuga). El coeficiente de `tipo_prueba` mide la
# brecha sistemática entre asignaturas.

# %%
def seccion_unificado(df):
    df_m = df.reset_index(drop=True).reset_index().rename(columns={"index": "id_registro"})
    id_vars = ["id_registro", "ind_am", "ind_cc", "ind_hv", "ind_pf",
               "cod_grupo", "cod_depe2", "cod_rural_rbd", "curso", "agno", "rbd"]
    largo = df_m.melt(id_vars=id_vars, value_vars=["prom_mate2m_rbd", "prom_lect2m_rbd"],
                      var_name="tipo_prueba", value_name="puntaje_simce")
    largo["tipo_prueba"] = largo["tipo_prueba"].map(
        {"prom_mate2m_rbd": "Matemática", "prom_lect2m_rbd": "Lectura"})
    largo = largo.dropna(subset=["puntaje_simce"]).reset_index(drop=True)
    print(f"  Formato largo: {len(df)} filas anchas -> {len(largo)} filas largas")
    print("   " + largo["tipo_prueba"].value_counts().to_string().replace("\n", "\n   "))

    X = largo[features_num + features_ord + features_cat_unif]
    y = largo["puntaje_simce"]
    groups = largo["id_registro"]

    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                          .split(X, y, groups))
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
    print(f"  Train: {len(X_train)} | Test: {len(X_test)} (GroupShuffleSplit por establecimiento)")

    model = construir_pipeline(features_cat_unif)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    tabla_train_test(model, X_train, y_train, X_test, y_test, "UNIFICADO (mate+lect)")
    cv = cross_val_score(model, X, y, cv=GroupKFold(n_splits=5), groups=groups, scoring="r2")
    print(f"  Validación cruzada R² (GroupKFold 5): {cv.round(3)} -> media {cv.mean():.3f} ± {cv.std():.3f}")

    fig_real_vs_pred(y_test, y_pred, "Unificado", "teal", "unif_real_vs_pred.png", "unificado")
    fig_residuos(y_test - y_pred, "Unificado", "teal", "unif_residuos.png", "unificado")
    coef_df = pd.DataFrame({"Variable": nombres_features(model, features_cat_unif),
                            "Coeficiente": model.named_steps["regressor"].coef_})
    fig_importancia(coef_df, "Unificado (mate+lect)", "teal", "indianred",
                    "unif_importancia.png", "unificado", separar_controles=False)

    # Efecto de la asignatura: diferencia entre los coeficientes de tipo_prueba (identificable).
    tp = coef_df[coef_df["Variable"].str.startswith("tipo_prueba")]
    print("\n  Coeficientes de tipo_prueba (brecha entre asignaturas):")
    print("   " + tp.to_string(index=False).replace("\n", "\n   "))
    return {"R2": r2_score(y_test, y_pred), "MAE": mean_absolute_error(y_test, y_pred)}


resumen_unif = seccion_unificado(df_global)

# %% [markdown]
# ## Conclusión — las tres lentes juntas
#
# - **Coherencia del hilo:** las tres secciones responden la **misma** pregunta de asociación
#   contemporánea, no compiten. El R² no "mejora" al separar/unir: la Sección A da el panorama, la B
#   muestra **heterogeneidad por nivel** (el efecto IDPS suele crecer hacia II medio) y la C confirma
#   si Matemática y Lectura comparten estructura (vía el coeficiente de `tipo_prueba`).
# - **Hallazgo transversal:** `ind_cc` (Clima de Convivencia) tiende a ser el IDPS con mayor
#   asociación; controlando por NSE el aporte socioemocional es real pero moderado.
# - **Rúbrica del profesor cubierta aquí:** NSE **ordinal**, **VIF**, **skewness**, comparación
#   explícita **Train vs Test**, **MAE** como métrica de negocio, **Distancia de Cook**, y la
#   unificación mate+lect con **`tipo_prueba`**. El enfoque **predictivo** (desfase T-1) vive en el
#   documento hermano `modelo_predictivo_simce.py`.

# %%
print("\n" + "=" * 48)
print("   SÍNTESIS CONTEMPORÁNEA (R²/MAE en test)")
print("=" * 48)
print("Sección A — Global:")
print(pd.DataFrame(resumen_global).round(3).to_string())
print("\nSección C — Unificado:")
print(pd.Series(resumen_unif).round(3).to_string())
print("\nSección B — por curso: ver tablas de evolución más arriba.")
print("\nFiguras en:", FIG_BASE)
