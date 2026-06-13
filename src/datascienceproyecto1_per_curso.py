# -*- coding: utf-8 -*-
"""
datascienceproyecto1_per_curso.py — Regresión lineal IDPS -> SIMCE, UN MODELO POR CURSO.

Variante per-curso del análisis contemporáneo de datascienceproyecto1.py. En vez de
un par de modelos (mate/lectura) sobre los 4 cursos mezclados, ajusta un par de
modelos POR CADA curso (4b, 6b, 8b, 2m) -> 8 modelos en total, para estudiar cómo
cambia el efecto de los IDPS según el nivel educativo (básica vs. media).

Diferencia metodológica clave respecto al script unificado:
  - El recorte de outliers (IQR) se calcula POR CURSO, sobre filas homogéneas, en vez
    de global sobre las ~98k filas mezcladas.
  - Dentro de un solo curso, 'curso' y la interacción NSE×curso son constantes y se
    omiten del modelo. Los predictores son los 4 IDPS + NSE (ordinal) + ruralidad +
    dependencia.

Entrada: data/processed/por_curso/dataset_historico_{curso}.csv
         (generados por build_dataset_historico_per_curso.py).
Salidas: figuras en reports/figuras_por_curso/ + tablas/comparaciones por consola.

Script autocontenido: duplica los helpers de datascienceproyecto1.py (ese módulo
corre todo su análisis al importarse, así que no es importable sin efectos).
Estructura OSEMN: Obtain -> Scrub -> Explore -> Model -> iNterpret, por curso.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventanas: los gráficos se guardan como PNG
import matplotlib.pyplot as plt

from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm


# =============================================================================
# Configuración y helpers (duplicados desde datascienceproyecto1.py)
# =============================================================================

# Carpeta NUEVA para no pisar las figuras del script unificado.
FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figuras_por_curso"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Orden de progresión educativa (básica -> media): eje X de los gráficos de evolución.
CURSOS = ["4b", "6b", "8b", "2m"]
CURSO_NOMBRE = {"4b": "4° básico", "6b": "6° básico", "8b": "8° básico", "2m": "II medio"}

# Orden jerárquico del NSE (menor a mayor), para el Ordinal Encoding.
orden_nse = ["Bajo", "Medio bajo", "Medio", "Medio alto", "Alto"]

# Predictores. 'curso'/'nse_x_curso' se omiten: son constantes dentro de cada curso.
features_num = ["ind_am", "ind_cc", "ind_hv", "ind_pf"]
features_ord = ["cod_grupo"]                      # NSE -> Ordinal Encoding
features_cat = ["cod_rural_rbd", "cod_depe2"]     # resto categórico -> One-Hot

# Indicadores IDPS con su nombre legible (para tablas y gráficos).
IDPS_NOMBRE = {
    "ind_am": "Autoestima Académica",
    "ind_cc": "Clima de Convivencia",
    "ind_hv": "Hábitos Saludables",
    "ind_pf": "Participación Ciudadana",
}

ASIGNATURAS = [("prom_mate2m_rbd", "Matemática"), ("prom_lect2m_rbd", "Lectura")]

# Mapas de estandarización categórica (no-op si ya vienen como texto canónico).
nse_map = {"1.0": "Bajo", "2.0": "Medio bajo", "3.0": "Medio",
           "4.0": "Medio alto", "5.0": "Alto"}
depe_map = {"1": "Municipal", "2": "Particular subvencionado",
            "3": "Particular pagado", "4": "SLEP"}
rural_map = {"1": "Urbano", "2": "Rural"}

_fig_contador = [0]


def guardar(nombre=None):
    """Guarda la figura actual como PNG en FIG_DIR y la cierra (sustituye a plt.show())."""
    _fig_contador[0] += 1
    if nombre is None:
        fig = plt.gcf()
        titulo = next((a.get_title() for a in fig.axes if a.get_title()), "")
        slug = re.sub(r"[^a-z0-9]+", "_", titulo.lower()).strip("_")[:40] or "figura"
        nombre = f"{_fig_contador[0]:02d}_{slug}.png"
    plt.savefig(FIG_DIR / nombre, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[figura guardada] {FIG_DIR / nombre}")


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


def construir_pipeline():
    """Pipeline: IDPS estandarizados + NSE ordinal + resto categórico One-Hot."""
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), features_num),
        ("ord", OrdinalEncoder(categories=[orden_nse],
                               handle_unknown="use_encoded_value", unknown_value=-1),
         features_ord),
        ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
    ])
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])


def nombres_features(model):
    """Nombres de las columnas tras el preprocesamiento, en el orden de los coeficientes."""
    pre = model.named_steps["preprocessor"]
    return (
        pre.named_transformers_["num"].get_feature_names_out(features_num).tolist()
        + features_ord
        + pre.named_transformers_["cat"].get_feature_names_out(features_cat).tolist()
    )


# =============================================================================
# 2) SCRUB: por curso (dropna + IQR calculado con los datos de ESE curso)
# =============================================================================

def scrub_curso(df, curso):
    df = df.copy()
    # Estandarización categórica (no-op inofensivo: ya vienen canónicas del build).
    df["cod_grupo"] = df["cod_grupo"].replace(nse_map)
    df["cod_depe2"] = df["cod_depe2"].replace(depe_map)
    df["cod_rural_rbd"] = df["cod_rural_rbd"].replace(rural_map)

    antes = len(df)
    df = df.dropna(subset=features_num + features_ord + features_cat
                   + ["prom_mate2m_rbd", "prom_lect2m_rbd"])
    print(f"  [{curso}] dropna: {antes} -> {len(df)} filas ({antes - len(df)} eliminadas)")

    # IQR POR CURSO sobre ambos targets (sobre filas homogéneas, no global).
    for col in ["prom_mate2m_rbd", "prom_lect2m_rbd"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        antes = len(df)
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
        print(f"  [{curso}] outliers IQR en {col}: {antes - len(df)} eliminados")

    print(f"  [{curso}] dataset final: {len(df)} filas")
    return df.reset_index(drop=True)


# =============================================================================
# 3) EXPLORE: skewness + VIF (tablas impresas, sin figura por curso)
# =============================================================================

def explorar_curso(df, curso):
    cols = ["prom_mate2m_rbd", "prom_lect2m_rbd"] + features_num
    print(f"\n  -- [{curso}] Skewness (|skew|>1 = asimetría marcada) --")
    tabla = df[cols].apply(lambda s: skew(s.dropna())).round(3).rename("skewness").to_frame()
    tabla["asimetria"] = tabla["skewness"].abs().apply(
        lambda v: "marcada" if v > 1 else ("moderada" if v > 0.5 else "leve"))
    print(tabla.to_string())

    print(f"\n  -- [{curso}] VIF de los IDPS --")
    vif = calcular_vif(df[features_num])
    print(vif.to_string(index=False))
    print(f"  Variables con VIF > 5: {vif.query('VIF > 5')['variable'].tolist() or 'ninguna'}")


# =============================================================================
# 4-5) MODEL + iNTERPRET: un modelo por (curso, asignatura)
# =============================================================================

def graficar_importancia(coef_df, curso, asignatura, color, nombre_fig):
    """Barras con el coeficiente de cada variable de interés (IDPS + contexto)."""
    orden = coef_df.reindex(coef_df["Coeficiente"].abs().sort_values(ascending=False).index)
    plt.figure(figsize=(9, 5))
    colores = [color if c > 0 else "gray" for c in orden["Coeficiente"]]
    plt.barh(orden["Variable"], orden["Coeficiente"], color=colores)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel(f"Coeficiente (efecto sobre SIMCE {asignatura})")
    plt.title(f"Importancia de variables — {asignatura} · {CURSO_NOMBRE[curso]}")
    plt.tight_layout()
    guardar(nombre_fig)


def graficar_real_vs_pred(y_test, y_pred, curso, asignatura, color, nombre_fig):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, s=12, color=color)
    lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", label="Predicción perfecta")
    plt.xlabel(f"Real (SIMCE {asignatura})")
    plt.ylabel("Predicho")
    plt.title(f"Real vs predicho — {asignatura} · {CURSO_NOMBRE[curso]}")
    plt.legend()
    plt.tight_layout()
    guardar(nombre_fig)


def cook_curso(df_train, target, curso, asignatura):
    """Distancia de Cook (tabla impresa): top establecimientos influyentes del train."""
    Xc = pd.DataFrame(index=df_train.index)
    Xc[features_num] = df_train[features_num]
    Xc["nse_ord"] = df_train["cod_grupo"].map({c: i + 1 for i, c in enumerate(orden_nse)})
    Xc = pd.concat([Xc, pd.get_dummies(df_train[features_cat], drop_first=True)], axis=1)
    Xc = sm.add_constant(Xc.astype(float))
    ols = sm.OLS(df_train[target].astype(float), Xc).fit()
    d = ols.get_influence().cooks_distance[0]
    umbral = 4 * d.mean()
    n_infl = int((d > umbral).sum())
    print(f"  [{curso}·{asignatura}] Cook: umbral 4·media(D)={umbral:.5f} -> "
          f"{n_infl} influyentes ({100 * n_infl / len(d):.1f}%)")
    influyentes = df_train.loc[d > umbral, ["rbd", "agno"]].assign(cook=d[d > umbral])
    print(influyentes.sort_values("cook", ascending=False).head(5)
          .to_string(index=False).replace("\n", "\n    "))


def modelar(df, curso, target, asignatura, color):
    """Ajusta un modelo (curso, asignatura) y devuelve coef IDPS + métricas de test."""
    X = df[features_num + features_ord + features_cat]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = construir_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Comparación explícita Train vs Test (overfitting).
    tt = pd.DataFrame([metricas_split(model, X_train, y_train),
                       metricas_split(model, X_test, y_test)],
                      index=["Train", "Test"]).round(3)
    tt.loc["Brecha (Test-Train)"] = (tt.loc["Test"] - tt.loc["Train"]).round(3)
    print(f"\n  RESULTADOS {asignatura} · {CURSO_NOMBRE[curso]} — Train vs Test:")
    print("    " + tt.to_string().replace("\n", "\n    "))

    # Coeficientes (IDPS en unidades de desviación estándar -> comparables entre cursos).
    coef_df = pd.DataFrame({"Variable": nombres_features(model),
                            "Coeficiente": model.named_steps["regressor"].coef_})

    # Figuras clave (streamlined).
    pref = f"{'mate' if 'mate' in target else 'lect'}_{curso}"
    graficar_importancia(coef_df, curso, asignatura, color, f"{pref}_importancia.png")
    graficar_real_vs_pred(y_test, y_pred, curso, asignatura, color, f"{pref}_real_vs_pred.png")

    # Cook como tabla impresa.
    cook_curso(df.loc[X_train.index], target, curso, asignatura)

    coef_idps = dict(zip(coef_df["Variable"], coef_df["Coeficiente"]))
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        **{ind: coef_idps[ind] for ind in features_num},
    }


# =============================================================================
# Comparación final entre cursos (el payoff)
# =============================================================================

def tabla_resumen(resultados, asignatura):
    """Tabla: filas = IDPS + R²/MAE, columnas = cursos."""
    filas = features_num + ["R2", "MAE"]
    data = {curso: [resultados[(curso, asignatura)][f] for f in filas] for curso in CURSOS}
    etiquetas = [IDPS_NOMBRE[f] for f in features_num] + ["R²", "MAE"]
    tabla = pd.DataFrame(data, index=etiquetas)[CURSOS].round(3)
    print(f"\n{'=' * 60}\n  RESUMEN COEFICIENTES IDPS POR CURSO — {asignatura}\n{'=' * 60}")
    print(tabla.to_string())
    return tabla


def graficar_evolucion(resultados, asignatura, color_base, nombre_fig):
    """Líneas: coeficiente de cada IDPS a lo largo de 4b -> 6b -> 8b -> 2m."""
    plt.figure(figsize=(9, 5.5))
    for ind in features_num:
        coefs = [resultados[(curso, asignatura)][ind] for curso in CURSOS]
        plt.plot([CURSO_NOMBRE[c] for c in CURSOS], coefs, marker="o", label=IDPS_NOMBRE[ind])
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Curso (progresión educativa)")
    plt.ylabel(f"Coeficiente sobre SIMCE {asignatura} (por desviación estándar)")
    plt.title(f"Evolución del efecto de los IDPS por curso — {asignatura}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    guardar(nombre_fig)


# =============================================================================
# Main
# =============================================================================

def main():
    color_asig = {"Matemática": "steelblue", "Lectura": "coral"}
    resultados = {}

    for curso in CURSOS:
        print(f"\n{'#' * 70}\n# CURSO {curso} ({CURSO_NOMBRE[curso]})\n{'#' * 70}")
        df = pd.read_csv(encontrar_por_curso(curso))
        print(f"  Cargado: {df.shape[0]} filas, años {sorted(df['agno'].unique())}")

        df = scrub_curso(df, curso)
        explorar_curso(df, curso)

        for target, asignatura in ASIGNATURAS:
            resultados[(curso, asignatura)] = modelar(
                df, curso, target, asignatura, color_asig[asignatura])

    # Comparación entre cursos.
    for target, asignatura in ASIGNATURAS:
        tabla_resumen(resultados, asignatura)
        graficar_evolucion(resultados, asignatura, color_asig[asignatura],
                           f"evolucion_idps_{'matematica' if asignatura == 'Matemática' else 'lectura'}.png")

    print("\nListo. Figuras en:", FIG_DIR)


if __name__ == "__main__":
    main()
