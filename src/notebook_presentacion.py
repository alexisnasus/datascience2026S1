# -*- coding: utf-8 -*-
"""notebook_presentacion.py — Presentación 2

Análisis del desempeño académico SIMCE mediante IDPS y variables contextuales.
Versión para presentar: código deduplicado y figuras enfocadas, manteniendo toda la
narrativa metodológica. Para subir a Colab: sube a /content los 5 CSV
(`dataset_historico_{4b,6b,8b,2m}.csv` y `dataset_maestro.csv`) y ejecuta todo.
"""

# %% [markdown]
# # Análisis del desempeño académico SIMCE mediante IDPS y variables contextuales
# ## Presentación 2 — Modelo final
#
# Este notebook estudia **cómo se asocian los Indicadores de Desarrollo Personal y
# Social (IDPS), junto con variables contextuales, con los resultados SIMCE de
# Matemática y Lectura según el nivel educativo evaluado**.
#
# ## Evolución metodológica respecto a la Presentación 1
#
# En la Presentación 1 se usó un primer modelo de regresión lineal múltiple sobre un
# dataset provisorio. Ya separaba Matemática y Lectura, pero no incorporaba el dataset
# final ni distinguía los cursos evaluados. A partir del feedback del profesor se
# realizaron ajustes importantes:
#
# - Se reformuló el enfoque para **no** presentar el modelo como predicción futura
#   cuando las variables IDPS y SIMCE pertenecen al mismo período.
# - Se probó una iteración intermedia que integraba Matemática y Lectura en una sola
#   variable objetivo usando `tipo_prueba` como variable explicativa.
# - Se mejoró la narrativa hacia un análisis **descriptivo-explicativo**, centrado en
#   asociaciones y no en causalidad.
# - Se incorporó la necesidad de comparar resultados, validar train/test y justificar
#   las métricas (MAE, RMSE, R²).
#
# Para esta entrega se adopta el enfoque por curso: **modelos separados por curso y
# asignatura**, lo que permite comparar si los IDPS se asocian de manera distinta con
# el SIMCE según el nivel escolar y según Matemática o Lectura.
#
# ## Pregunta de investigación
#
# ¿Qué relación existe entre los IDPS, las variables contextuales de los establecimientos
# y los puntajes SIMCE de Matemática y Lectura, considerando diferencias entre cursos?
#
# ## Objetivo general
#
# Analizar la asociación entre IDPS, variables contextuales y resultados SIMCE por
# establecimiento, diferenciando por curso y asignatura, para identificar qué variables
# se relacionan con mayores o menores puntajes en Matemática y Lectura.
#
# ## Enfoque OSEMN
#
# 1. **Obtain:** carga del dataset final por curso.
# 2. **Scrub:** revisión de nulos, categorías, rangos y outliers por curso.
# 3. **Explore:** análisis descriptivo y correlaciones.
# 4. **Model:** regresión lineal múltiple por curso y asignatura.
# 5. **iNterpret:** comparación de métricas, coeficientes e interpretación educativa.
#
# > El enfoque principal es **contemporáneo**: estudia asociaciones del mismo período, no
# > afirma causalidad ni predicción futura. Al final se agrega un anexo predictivo con
# > desfase temporal explícito (T-1) para mostrar predicción real.

# %% [code] 0) Imports y configuración
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display
except ImportError:                      # permite correr como script fuera de Colab
    display = print

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 100)

DATA_DIR = Path("/content")              # en Colab los CSV van a /content
CURSOS = ["4b", "6b", "8b", "2m"]
CURSO_NOMBRE = {"4b": "4° básico", "6b": "6° básico", "8b": "8° básico", "2m": "II medio"}
IDPS_NOMBRE = {"ind_am": "Autoestima", "ind_cc": "Convivencia",
               "ind_hv": "Hábitos", "ind_pf": "Participación"}
ASIGNATURAS = [("prom_mate2m_rbd", "Matemática"), ("prom_lect2m_rbd", "Lectura")]
PALETA = {"Matemática": "steelblue", "Lectura": "coral"}
FEATURES_IDPS = ["ind_am", "ind_cc", "ind_hv", "ind_pf"]
TARGETS = ["prom_mate2m_rbd", "prom_lect2m_rbd"]


# %% [markdown]
# # 1) OBTAIN — Carga de datos finales
#
# Se usan los cuatro datasets por curso consolidados desde el pipeline del proyecto
# (`dataset_historico_{4b,6b,8b,2m}.csv`). Cada archivo contiene puntajes SIMCE, IDPS y
# variables contextuales por establecimiento (`rbd`), año (`agno`) y curso (`curso`).

# %% [code] 1.1) Carga
def cargar_datos(data_dir=DATA_DIR):
    rutas = {c: data_dir / f"dataset_historico_{c}.csv" for c in CURSOS}
    faltan = [str(p) for p in rutas.values() if not p.exists()]
    if faltan:                            # fallback: buscar bajo /content y el cwd
        for raiz in [Path.cwd(), Path("/content")]:
            hit = next(raiz.rglob("dataset_historico_4b.csv"), None) if raiz.exists() else None
            if hit:
                return cargar_datos(hit.parent)
        raise FileNotFoundError(f"No encontré los CSV por curso. Faltan: {faltan}")
    partes = []
    for c in CURSOS:
        t = pd.read_csv(rutas[c])
        print(f"  dataset_historico_{c}.csv: {t.shape[0]:,} filas × {t.shape[1]} columnas")
        partes.append(t)
    return pd.concat(partes, ignore_index=True)


df_raw = cargar_datos()
print(f"\nDataset unido: {df_raw.shape[0]:,} filas × {df_raw.shape[1]} columnas")
display(df_raw.head())

# %% [code] 1.2) Panorama inicial (registros, años, cobertura)
print("Años disponibles:", sorted(df_raw["agno"].dropna().unique()))
print("\nRegistros por curso:")
display(df_raw["curso"].astype(str).str.strip().value_counts()
        .reindex(CURSOS).rename(index=CURSO_NOMBRE).to_frame("n_registros"))
print("Registros por año y curso:")
display(pd.crosstab(df_raw["agno"], df_raw["curso"].astype(str).str.strip()).reindex(columns=CURSOS))


# %% [markdown]
# # 2) SCRUB — Limpieza y preparación
#
# El dataset ya viene consolidado. Aquí se estandarizan categorías (NSE, dependencia,
# ruralidad), se revisan nulos y rangos, y se tratan outliers por curso. También se
# codifica el NSE como variable **ordinal**, respondiendo al feedback de la Presentación 1.
#
# ## Nota: codificación ordinal del NSE
#
# `cod_grupo` representa el nivel socioeconómico del establecimiento y tiene un orden
# natural: `Bajo < Medio bajo < Medio < Medio alto < Alto`. Por eso se crea `nse_ord`
# (1 a 5) respetando ese orden. Es la misma idea que `OrdinalEncoder`, pero explícita y
# fácil de defender. En el modelo `nse_ord` se estandariza junto con los IDPS, así su
# coeficiente es comparable en magnitud con ellos (cambio en SIMCE por una desviación
# estándar de NSE).

# %% [code] 2.1) Normalización de categorías y NSE ordinal
NSE_MAP = {**{k: "Bajo" for k in ["1", "1.0", "Bajo"]},
           **{k: "Medio bajo" for k in ["2", "2.0", "Medio bajo"]},
           **{k: "Medio" for k in ["3", "3.0", "Medio"]},
           **{k: "Medio alto" for k in ["4", "4.0", "Medio alto"]},
           **{k: "Alto" for k in ["5", "5.0", "Alto"]}}
NSE_ORD = {"Bajo": 1, "Medio bajo": 2, "Medio": 3, "Medio alto": 4, "Alto": 5}
COLS_REQ = ["rbd", "agno", "curso", *TARGETS, "cod_grupo", "nse_ord",
            "cod_rural_rbd", "cod_depe2", *FEATURES_IDPS]


def preparar(df):
    df = df.copy()
    for c in ["cod_grupo", "cod_depe2", "cod_rural_rbd", "curso"]:
        df[c] = df[c].astype(str).str.strip()
    df["cod_grupo"] = df["cod_grupo"].replace(NSE_MAP)
    df["nse_ord"] = df["cod_grupo"].map(NSE_ORD)
    for c in ["rbd", "agno", *TARGETS, *FEATURES_IDPS, "nse_ord"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["curso_nombre"] = df["curso"].map(CURSO_NOMBRE)
    return df


df_prep = preparar(df_raw)
print("Categorías detectadas:")
for col in ["cod_grupo", "cod_rural_rbd", "cod_depe2", "curso"]:
    print(f"- {col}: {sorted(df_prep[col].dropna().unique())}")

print("\nNulos por columna relevante:")
display(df_prep[COLS_REQ].isna().sum().to_frame("nulos"))

df_scrub = df_prep.dropna(subset=COLS_REQ).copy()
print(f"Filas: {len(df_prep):,} → tras dropna: {len(df_scrub):,} "
      f"({len(df_prep) - len(df_scrub):,} eliminadas)")

# %% [code] 2.2) Revisión de rangos por curso
print("Rangos (min / mean / max) de puntajes e IDPS por curso:")
display(df_scrub.groupby("curso")[TARGETS + FEATURES_IDPS]
        .agg(["min", "mean", "max"]).reindex(CURSOS).rename(index=CURSO_NOMBRE).round(2))

# %% [markdown]
# ## Nota: análisis de simetría (skewness)
#
# El feedback de la Presentación 1 pidió revisar la simetría de las variables. Regla de
# lectura: `|s|<0.5` aprox. simétrica · `0.5≤|s|<1` asimetría moderada · `|s|≥1` alta.
# No cambia el modelo automáticamente, pero justifica si las variables son razonables
# para una regresión lineal.

# %% [code] 2.3) Skewness por curso
skew_tbl = (df_scrub.groupby("curso")[TARGETS + FEATURES_IDPS].skew()
            .reindex(CURSOS).rename(index=CURSO_NOMBRE).round(3))
print("Skewness por curso (puntajes e IDPS):")
display(skew_tbl)

# %% [markdown]
# ## Nota: outliers por IQR calculado dentro de cada curso
#
# El rango intercuartílico (IQR) se calcula **por curso** porque mezclar 4° básico, 6°,
# 8° e II medio puede marcar como outlier un valor que solo pertenece a otra escala de
# comparación educativa. Se filtran los puntajes SIMCE fuera de `[Q1−1.5·IQR, Q3+1.5·IQR]`.

# %% [code] 2.4) Filtro de outliers IQR por curso
def quitar_outliers_iqr(df):
    keep = pd.Series(True, index=df.index)
    for _, g in df.groupby("curso"):
        kc = pd.Series(True, index=g.index)
        for col in TARGETS:
            q1, q3 = g[col].quantile(0.25), g[col].quantile(0.75)
            iqr = q3 - q1
            kc &= g[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        keep.loc[g.index] = kc
    return df.loc[keep].copy()


df_modelo = quitar_outliers_iqr(df_scrub)
print(f"Filas: {len(df_scrub):,} → tras IQR: {len(df_modelo):,} "
      f"({len(df_scrub) - len(df_modelo):,} eliminadas)")
display(df_modelo.groupby("curso_nombre").size().reindex(
    [CURSO_NOMBRE[c] for c in CURSOS]).to_frame("registros_finales"))

# %% [markdown]
# ## Multicolinealidad entre IDPS (VIF)
#
# VIF<5 indica que no hay redundancia grave entre los IDPS; aun así los coeficientes
# individuales se interpretan con cautela.

# %% [code] 2.5) VIF por curso
def vif(df_num):
    filas = []
    for col in df_num.columns:
        X = df_num.drop(columns=col)
        r2 = LinearRegression().fit(X, df_num[col]).score(X, df_num[col])
        filas.append({"variable": IDPS_NOMBRE.get(col, col),
                      "VIF": np.inf if 1 - r2 < 1e-12 else 1 / (1 - r2)})
    return pd.DataFrame(filas)


vif_tbl = pd.concat([vif(df_modelo[df_modelo["curso"] == c][FEATURES_IDPS]).assign(
    curso=CURSO_NOMBRE[c]) for c in CURSOS], ignore_index=True)
print("VIF de los IDPS por curso:")
display(vif_tbl.pivot(index="variable", columns="curso", values="VIF")
        [[CURSO_NOMBRE[c] for c in CURSOS]].round(2))


# %% [markdown]
# # 3) EXPLORE — Análisis descriptivo
#
# Se compara la distribución de SIMCE entre cursos, la correlación global entre SIMCE,
# IDPS y NSE, y la correlación IDPS↔SIMCE dentro de cada curso. Esto verifica si tiene
# sentido modelar cada nivel por separado.

# %% [code] 3.1) Distribución de SIMCE por curso y matriz de correlación global
plot_df = df_modelo.melt(id_vars=["curso_nombre"], value_vars=TARGETS,
                         var_name="_c", value_name="puntaje")
plot_df["asignatura"] = plot_df["_c"].map(dict(ASIGNATURAS))
plot_df["curso_nombre"] = pd.Categorical(
    plot_df["curso_nombre"], [CURSO_NOMBRE[c] for c in CURSOS], ordered=True)

plt.figure(figsize=(10, 4.5))
sns.boxplot(data=plot_df, x="curso_nombre", y="puntaje", hue="asignatura", palette=PALETA)
plt.title("Distribución de puntajes SIMCE por curso y asignatura")
plt.xlabel("Curso"); plt.ylabel("Puntaje SIMCE"); plt.tight_layout(); plt.show()

corr_labels = {"prom_mate2m_rbd": "Matemática", "prom_lect2m_rbd": "Lectura",
               "ind_am": "Autoestima", "ind_cc": "Convivencia", "ind_hv": "Hábitos",
               "ind_pf": "Participación", "nse_ord": "NSE"}
plt.figure(figsize=(8, 6))
sns.heatmap(df_modelo[list(corr_labels)].corr().rename(index=corr_labels, columns=corr_labels),
            annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlación global: SIMCE, IDPS y NSE"); plt.tight_layout(); plt.show()

# %% [code] 3.2) Correlación IDPS ↔ SIMCE por curso (tablas)
filas_corr = []
for curso in CURSOS:
    g = df_modelo[df_modelo["curso"] == curso]
    for idps in FEATURES_IDPS:
        for target, asig in ASIGNATURAS:
            filas_corr.append({"indicador": IDPS_NOMBRE[idps], "asignatura": asig,
                               "curso": CURSO_NOMBRE[curso], "corr": g[idps].corr(g[target])})
corr_idps_simce = pd.DataFrame(filas_corr)
for asig in ["Matemática", "Lectura"]:
    print(f"Correlación IDPS ↔ SIMCE — {asig}")
    display(corr_idps_simce[corr_idps_simce["asignatura"] == asig]
            .pivot(index="indicador", columns="curso", values="corr")
            [[CURSO_NOMBRE[c] for c in CURSOS]].round(3))


# %% [markdown]
# # 4) MODEL — Regresión lineal múltiple por curso y asignatura
#
# Se entrenan **8 modelos** (4 cursos × 2 asignaturas: Matemática y Lectura). Este
# enfoque permite comparar directamente los coeficientes de IDPS entre niveles y
# asignaturas.
#
# ## Variables del modelo
# **Targets:** `prom_mate2m_rbd` (Matemática), `prom_lect2m_rbd` (Lectura).
# **Predictores:** IDPS (`ind_am`, `ind_cc`, `ind_hv`, `ind_pf`) + NSE ordinal
# estandarizado (`nse_ord`) + ruralidad (`cod_rural_rbd`) + dependencia (`cod_depe2`).
# Los IDPS y `nse_ord` se estandarizan para que sus coeficientes sean comparables;
# ruralidad y dependencia van con one-hot (coeficientes = diferencia vs categoría base).
#
# ## Decisiones metodológicas de esta iteración
# Respecto al primer modelo, se ajusta la validación para hacerla más defendible:
# - **Split agrupado por `rbd`:** un mismo establecimiento no aparece a la vez en train y test.
# - **Validación cruzada agrupada (GroupKFold):** evalúa estabilidad respetando la unidad establecimiento.
# - **NSE ordinal estandarizado:** permite comparar su magnitud con los IDPS.
# - **One-hot con categoría base (`drop="first"`):** evita redundancia entre categorías.

# %% [code] 4.1) Funciones de modelado (reutilizables: principal y sin NSE)
def metricas(yr, yp):
    return {"MAE": mean_absolute_error(yr, yp),
            "RMSE": np.sqrt(mean_squared_error(yr, yp)), "R2": r2_score(yr, yp)}


def pipeline(feat_num, feat_cat):
    pre = ColumnTransformer([("num", StandardScaler(), feat_num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), feat_cat)])
    return Pipeline([("pre", pre), ("reg", LinearRegression())])


def correr_por_curso(feat_num, feat_cat):
    """Devuelve (métricas, coeficientes) para los 8 modelos curso×asignatura."""
    met, coef = [], []
    for curso in CURSOS:
        d = df_modelo[df_modelo["curso"] == curso]
        for target, asig in ASIGNATURAS:
            data = d.dropna(subset=feat_num + feat_cat + [target, "rbd"])
            X, y, g = data[feat_num + feat_cat], data[target], data["rbd"]
            tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))
            m = pipeline(feat_num, feat_cat).fit(X.iloc[tr], y.iloc[tr])
            mtr = metricas(y.iloc[tr], m.predict(X.iloc[tr]))
            mte = metricas(y.iloc[te], m.predict(X.iloc[te]))
            r2cv = cross_validate(pipeline(feat_num, feat_cat), X, y, groups=g,
                                  cv=GroupKFold(5), scoring="r2")["test_score"].mean()
            met.append({"curso": CURSO_NOMBRE[curso], "asignatura": asig, "n": len(data),
                        "MAE_train": mtr["MAE"], "MAE_test": mte["MAE"],
                        "RMSE_test": mte["RMSE"], "R2_train": mtr["R2"],
                        "R2_test": mte["R2"], "R2_CV": r2cv})
            nombres = (m.named_steps["pre"].named_transformers_["num"].get_feature_names_out(feat_num).tolist()
                       + m.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(feat_cat).tolist())
            for v, c in zip(nombres, m.named_steps["reg"].coef_):
                coef.append({"curso": CURSO_NOMBRE[curso], "asignatura": asig, "variable": v, "coef": c})
    return pd.DataFrame(met), pd.DataFrame(coef)


FEAT_NUM = ["ind_am", "ind_cc", "ind_hv", "ind_pf", "nse_ord"]
FEAT_CAT = ["cod_rural_rbd", "cod_depe2"]
met_principal, coef_principal = correr_por_curso(FEAT_NUM, FEAT_CAT)
orden = [CURSO_NOMBRE[c] for c in CURSOS]
met_principal["curso"] = pd.Categorical(met_principal["curso"], orden, ordered=True)
met_principal = met_principal.sort_values(["curso", "asignatura"]).reset_index(drop=True)

print("Métricas de los 8 modelos (train / test / validación cruzada):")
display(met_principal.round(3))

# %% [code] 4.2) Desempeño por curso (R² y MAE) y coeficientes IDPS
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, metr, titulo in zip(axes, ["R2_test", "MAE_test"], ["R² test", "MAE test"]):
    sns.barplot(data=met_principal, x="curso", y=metr, hue="asignatura", palette=PALETA, ax=ax)
    ax.set_title(titulo); ax.set_xlabel("Curso"); ax.tick_params(axis="x", rotation=15)
plt.suptitle("Desempeño por curso y asignatura"); plt.tight_layout(); plt.show()

coef_idps = coef_principal[coef_principal["variable"].isin(FEATURES_IDPS)].copy()
coef_idps["indicador"] = coef_idps["variable"].map(IDPS_NOMBRE)
coef_idps["curso"] = pd.Categorical(coef_idps["curso"], orden, ordered=True)
for asig in ["Matemática", "Lectura"]:
    print(f"Coeficientes IDPS estandarizados — {asig}")
    display(coef_idps[coef_idps["asignatura"] == asig]
            .pivot(index="indicador", columns="curso", values="coef")
            [[CURSO_NOMBRE[c] for c in CURSOS]].round(3))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, asig in zip(axes, ["Matemática", "Lectura"]):
    sns.lineplot(data=coef_idps[coef_idps["asignatura"] == asig], x="curso", y="coef",
                 hue="indicador", marker="o", ax=ax)
    ax.axhline(0, color="black", lw=0.8); ax.set_title(f"Coeficientes IDPS — {asig}")
    ax.set_xlabel("Curso"); ax.set_ylabel("Coef. estandarizado"); ax.tick_params(axis="x", rotation=15)
plt.suptitle("Asociación IDPS→SIMCE por curso"); plt.tight_layout(); plt.show()

# %% [code] 4.3) Importancia de variables por modelo (todas las variables)
fig, axes = plt.subplots(4, 2, figsize=(14, 18))
for i, curso in enumerate(CURSOS):
    for j, (target, asig) in enumerate(ASIGNATURAS):
        ax = axes[i, j]
        c = coef_principal[(coef_principal["curso"] == CURSO_NOMBRE[curso])
                           & (coef_principal["asignatura"] == asig)].copy()
        c = c.reindex(c["coef"].abs().sort_values(ascending=False).index)
        colores = [PALETA[asig] if v > 0 else "gray" for v in c["coef"]]
        ax.barh(c["variable"], c["coef"], color=colores)
        ax.axvline(0, color="black", lw=0.8); ax.invert_yaxis()
        ax.set_title(f"{asig} · {CURSO_NOMBRE[curso]}")
plt.suptitle("Importancia de variables (modelo principal)", y=1.001)
plt.tight_layout(); plt.show()


# %% [markdown]
# ## 4.4) Variante sin NSE — ¿cuánto del "efecto IDPS" es en realidad NSE?
#
# En Chile el nivel socioeconómico domina parte importante de las diferencias
# educativas, por lo que puede ocultar la lectura de los IDPS. Se reentrenan los 8
# modelos quitando solo `nse_ord`. Si el R² baja, el NSE aportaba información real; si
# los coeficientes IDPS suben, parte de su asociación estaba mezclada con el contexto
# socioeconómico.

# %% [code] 4.4) Comparación con / sin NSE
met_sin_nse, coef_sin_nse = correr_por_curso(FEATURES_IDPS, FEAT_CAT)
comp = met_principal[["curso", "asignatura", "R2_test", "MAE_test"]].merge(
    met_sin_nse[["curso", "asignatura", "R2_test", "MAE_test"]],
    on=["curso", "asignatura"], suffixes=("_con_NSE", "_sin_NSE"))
comp["delta_R2"] = comp["R2_test_sin_NSE"] - comp["R2_test_con_NSE"]
comp["delta_MAE"] = comp["MAE_test_sin_NSE"] - comp["MAE_test_con_NSE"]
print("Impacto de quitar NSE (test):")
display(comp.round(3))

plt.figure(figsize=(9, 4.5))
sns.barplot(data=comp, x="curso", y="delta_R2", hue="asignatura", palette=PALETA)
plt.axhline(0, color="black", lw=0.8)
plt.title("Caída de R² al quitar NSE (más negativo = NSE aportaba más)")
plt.xlabel("Curso"); plt.ylabel("Δ R² (sin − con)"); plt.tight_layout(); plt.show()


# %% [markdown]
# # 5) Modelos complementarios (responden el feedback del profesor)
#
# - **Global:** todos los cursos juntos (`curso` como control de escala) → panorama y línea base.
# - **Unificado `tipo_prueba`:** Matemática y Lectura apiladas en una sola variable objetivo;
#   el coeficiente de `tipo_prueba` mide la brecha sistemática entre asignaturas (lo pidió el profe).

# %% [code] 5) Global + unificado
print("MODELO GLOBAL (todos los cursos):")
for target, asig in ASIGNATURAS:
    fn, fc = FEAT_NUM, ["cod_rural_rbd", "cod_depe2", "curso"]
    data = df_modelo.dropna(subset=fn + fc + [target, "rbd"])
    X, y, g = data[fn + fc], data[target], data["rbd"]
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))
    m = pipeline(fn, fc).fit(X.iloc[tr], y.iloc[tr]); mte = metricas(y.iloc[te], m.predict(X.iloc[te]))
    print(f"  {asig:11s} R²_test={mte['R2']:.3f}  MAE_test={mte['MAE']:.2f}")

du = df_modelo.melt(id_vars=["rbd", "curso", "nse_ord", "cod_rural_rbd", "cod_depe2", *FEATURES_IDPS],
                    value_vars=TARGETS, var_name="_c", value_name="puntaje")
du["tipo_prueba"] = du["_c"].map(dict(ASIGNATURAS))
du = du.dropna(subset=["puntaje"])
fn, fc = FEAT_NUM, ["cod_rural_rbd", "cod_depe2", "curso", "tipo_prueba"]
X, y, g = du[fn + fc], du["puntaje"], du["rbd"]
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))
m = pipeline(fn, fc).fit(X.iloc[tr], y.iloc[tr]); mte = metricas(y.iloc[te], m.predict(X.iloc[te]))
nombres = (m.named_steps["pre"].named_transformers_["num"].get_feature_names_out(fn).tolist()
           + m.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(fc).tolist())
coef_tp = {v: c for v, c in zip(nombres, m.named_steps["reg"].coef_) if v.startswith("tipo_prueba")}
print(f"\nMODELO UNIFICADO (tipo_prueba): R²_test={mte['R2']:.3f}  MAE_test={mte['MAE']:.2f}")
print(f"  coef {list(coef_tp)[0]} = {list(coef_tp.values())[0]:.2f} "
      f"(Matemática vs Lectura, a igualdad del resto)")


# %% [markdown]
# # 6) Anexo predictivo — desfase temporal T-1 (out-of-time)
#
# El análisis principal es contemporáneo (asociación). Este anexo hace predicción
# **real**: predictores en el año **T-1**, target SIMCE en **T**, y test = 2025 (nunca
# visto en el ajuste). Se comparan **Solo IDPS** vs **Autorregresivo** (agrega el SIMCE
# previo). Por el desfase estricto solo sobreviven los cursos evaluados todos los años
# (`4b` y `2m`). Requiere `dataset_maestro.csv` subido a /content.

# %% [code] 6) Predictivo T-1
def cargar_maestro(data_dir=DATA_DIR):
    p = data_dir / "dataset_maestro.csv"
    if p.exists():
        return pd.read_csv(p)
    for raiz in [Path.cwd(), Path("/content")]:
        hit = next(raiz.rglob("dataset_maestro.csv"), None) if raiz.exists() else None
        if hit:
            return pd.read_csv(hit)
    raise FileNotFoundError("Sube dataset_maestro.csv a /content.")


dp = cargar_maestro()
FC = ["cod_rural_rbd", "cod_depe2"]
FI = ["idps_am_prev", "idps_cc_prev", "idps_hv_prev", "idps_pf_prev", "nse_ord", "curso_ord"]
FA = FI + ["simce_mate_prev", "simce_lect_prev"]
TG = {"Matemática": "target_mate", "Lectura": "target_lect"}
dp = dp.dropna(subset=list(dict.fromkeys(FA + FC + list(TG.values()) + ["agno"])))
train, test = dp[dp["agno"].isin([2017, 2018, 2023, 2024])], dp[dp["agno"] == 2025]
print(f"Predictivo T-1 | train {len(train):,} | test 2025 {len(test):,} | cursos {sorted(dp['curso'].unique())}")

filas = []
for asig, tg in TG.items():
    for nombre, fn in [("Solo IDPS", FI), ("Autorregresivo", FA)]:
        m = pipeline(fn, FC).fit(train[fn + FC], train[tg])
        mte = metricas(test[tg], m.predict(test[fn + FC]))
        filas.append({"asignatura": asig, "modelo": nombre,
                      "MAE_test": mte["MAE"], "R2_test": mte["R2"]})
print("Predictivo T-1 — resultados sobre 2025:")
display(pd.DataFrame(filas).round(3))


# %% [markdown]
# # 7) iNTERPRET — Síntesis e interpretación
#
# Se conectan los resultados con la pregunta del proyecto y con la rúbrica.

# %% [code] 7.1) Síntesis cuantitativa
print("Mejor → peor R² test (modelo principal):")
display(met_principal.sort_values("R2_test", ascending=False)
        [["curso", "asignatura", "MAE_test", "RMSE_test", "R2_test"]].round(3).reset_index(drop=True))

br = met_principal[["curso", "asignatura", "MAE_train", "MAE_test", "R2_train", "R2_test", "R2_CV"]].copy()
br["brecha_R2_train_test"] = (br["R2_train"] - br["R2_test"]).round(3)
print("Brecha train–test y comparación con validación cruzada (estabilidad):")
display(br.round(3))

# %% [markdown]
# ## Conclusiones (números reales de esta corrida)
#
# **1. El NSE es el predictor dominante y su peso crece con el nivel.** Estandarizado,
# pasa de ~10 pts en 4° básico a ~28 pts en II medio: 2–4× el IDPS más influyente.
#
# **2. Quitar el NSE empeora el ajuste, y el deterioro escala con el curso:** Δ R² ≈ −0,08
# en básica hasta **−0,19 en II medio**. Buena parte del "efecto IDPS" es, en realidad, NSE.
#
# **3. Entre los IDPS, Convivencia (`ind_cc`) es el más estable y dominante.** Participación
# (`ind_pf`) es inestable / cambia de signo → no interpretar como efecto.
#
# **4. El ajuste mejora con el nivel educativo:** R² test ~0,37–0,43 en básica vs **0,61
# (mate) / 0,56 (lect) en II medio**. MAE entre 13 y 19 puntos SIMCE.
#
# **5. Sin sobreajuste:** en los 8 modelos R²_test ≈ R²_CV y la brecha train–test es pequeña.
#
# **6. `tipo_prueba` ≈ −1,4:** la brecha mate–lectura es pequeña → comparten estructura,
# lo que justifica el modelo unificado.
#
# **7. Predictivo T-1:** el predictor fuerte del SIMCE futuro es el **SIMCE previo**
# (R² 0,62/0,52); los IDPS solos anticipan ~0,36–0,38. Predicen algo, pero no son el motor.
#
# **Métrica elegida — MAE (principal):** error en puntos SIMCE reales, interpretable para
# directivos. R²: bondad de ajuste comparable entre cursos. RMSE: vigila errores grandes.

# %% [markdown]
# ## Comparación de iteraciones
#
# | Iteración | Qué hacía | Limitación principal | Qué se mejora en el modelo final |
# |---|---|---|---|
# | Primer modelo (Presentación 1) | Regresión lineal para Matemática y Lectura por separado sobre dataset provisorio. | No incorpora cursos ni dataset final consolidado. | Se usa el dataset final por curso y se comparan 8 modelos. |
# | Iteración intermedia (unificada) | Integra Matemática y Lectura en formato largo con `tipo_prueba`. Mejora la narrativa descriptivo-explicativa. | El modelo unificado dificulta comparar cómo cambian los IDPS por asignatura y curso. | Se separa por curso y asignatura para interpretar coeficientes con claridad (y se conserva el unificado como complemento). |
# | Modelo final (Presentación 2) | Regresión lineal múltiple por curso y asignatura + complementos (global, `tipo_prueba`, predictivo T-1). | Sigue siendo contemporáneo en su parte principal; no prueba causalidad. | Responde mejor la pregunta educativa y la rúbrica: iteración, validación, métricas y un anexo predictivo real. |
#
# ## Justificación del modelo final
# Se elige el enfoque de **8 regresiones lineales múltiples** porque: mantiene la
# interpretabilidad de los coeficientes; evita mezclar niveles educativos en un solo
# ajuste; permite comparar Matemática vs Lectura; y permite observar si la asociación
# IDPS↔SIMCE cambia entre 4° básico e II medio. Los modelos global, unificado y
# predictivo T-1 lo complementan respondiendo, uno a uno, las observaciones del profesor.
#
# ## Limitaciones
# - Los coeficientes representan **asociaciones, no causalidad**.
# - El análisis principal es **contemporáneo** (IDPS y SIMCE del mismo año).
# - Para afirmar predicción futura se usa el anexo T-1 (IDPS en T-1 → SIMCE en T).
# - Los IDPS presentan multicolinealidad leve → interpretar coeficientes con cautela.
# - No todos los cursos tienen los mismos años de SIMCE evaluados.
