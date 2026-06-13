#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset_historico.py — Consolidación final (etapa de join CONTEMPORÁNEO).

Une los dos datasets consolidados (IDPS y SIMCE) en un único dataset analítico
con estructura CONTEMPORÁNEA (mismo año), estilo `dataset_historico_final.csv`
pero incluyendo TODOS los cursos (2m, 4b, 6b, 8b) en vez de solo 2° medio.

    Predictores y target del MISMO año T  (análisis descriptivo-explicativo).

Es la entrada de los modelos descriptivos (regresión lineal / Ridge) que estudian
la relación contemporánea IDPS <-> SIMCE. Para el enfoque PREDICTIVO con desfase
temporal, ver `build_dataset_maestro.py` (dataset_maestro.csv).

Salida: data/processed/dataset_historico_completo.csv (una fila por (rbd, agno, curso)).

Las rutas se resuelven desde __file__, así que el script corre desde cualquier CWD.
NO imputa, NO escala y NO elimina outliers/nulos: esas decisiones (Scrub de
modelado) viven en el notebook/script de modelado.
"""
from pathlib import Path

import pandas as pd

# --- Rutas ---  (src/construir_datasets/<archivo>.py -> raíz del repo = parents[2])
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed"
IDPS_CSV = PROCESSED / "dataset_consolidado_idps.csv"
SIMCE_CSV = PROCESSED / "dataset_simce_consolidado.csv"
OUT_CSV = PROCESSED / "dataset_historico_completo.csv"

KEY = ["rbd", "agno", "curso"]

# --- Mapeos de normalización ---------------------------------------------------
# Las fuentes mezclan, según el año, códigos numéricos ('1', '1.0', ...) y
# etiquetas de texto ('Bajo', 'Urbano', ...). Todo se normaliza con _norm_key.

# NSE (cod_grupo) -> ordinal 1..5 respetando la jerarquía socioeconómica.
NSE_ORDINAL = {
    "1": 1, "1.0": 1, "bajo": 1,
    "2": 2, "2.0": 2, "medio bajo": 2,
    "3": 3, "3.0": 3, "medio": 3,
    "4": 4, "4.0": 4, "medio alto": 4,
    "5": 5, "5.0": 5, "alto": 5,
}
# Etiqueta canónica del NSE a partir del ordinal (texto legible para los modelos).
NSE_LABEL = {1: "Bajo", 2: "Medio bajo", 3: "Medio", 4: "Medio alto", 5: "Alto"}

RURAL_LABEL = {
    "1": "Urbano", "urbano": "Urbano",
    "2": "Rural", "rural": "Rural",
}
DEPE_LABEL = {
    "1": "Municipal", "municipal": "Municipal",
    "2": "Particular subvencionado", "particular subvencionado": "Particular subvencionado",
    "3": "Particular pagado", "particular pagado": "Particular pagado",
    "4": "SLEP", "slep": "SLEP",
}
CURSOS_VALIDOS = {"2m", "4b", "6b", "8b"}

# Puntaje SIMCE mínimo plausible. La escala real va de ~100 a ~400; valores en 0
# (o muy bajos) son errores de digitalización de la fuente, NO ceros reales, y se
# descartan como dato inválido (distinto del recorte estadístico IQR del modelo).
UMBRAL_SIMCE_MIN = 100


def _norm_key(valor):
    """Normaliza un valor a una clave comparable: minúsculas y sin espacios."""
    return str(valor).strip().lower()


def _mapear(serie, mapa, *, conservar_original=False):
    """Mapea una serie usando _norm_key. NaN se mantiene como NaN.

    Si conservar_original es True, los valores no reconocidos se dejan tal cual;
    si no, se convierten en NaN.
    """
    def f(v):
        if pd.isna(v):
            return None
        clave = _norm_key(v)
        if clave in mapa:
            return mapa[clave]
        return v if conservar_original else None
    return serie.map(f)


def cargar():
    idps = pd.read_csv(IDPS_CSV, low_memory=False)
    simce = pd.read_csv(SIMCE_CSV, low_memory=False)
    return idps, simce


def limpiar_claves(df):
    """Homologa las claves (rbd int, agno int, curso válido) en ambos datasets."""
    df = df.copy()
    df = df.dropna(subset=["rbd"])
    df["rbd"] = df["rbd"].astype(float).astype(int)
    df["agno"] = df["agno"].astype(int)
    df["curso"] = df["curso"].astype(str).str.strip()
    df = df[df["curso"].isin(CURSOS_VALIDOS)]
    return df


def construir_historico(idps, simce):
    idps = limpiar_claves(idps)
    simce = limpiar_claves(simce)

    # Evita filas duplicadas por clave (los consolidados ya deberían ser únicos).
    idps = idps.drop_duplicates(subset=KEY)
    simce = simce.drop_duplicates(subset=KEY)

    # --- SIMCE (puntajes + contexto), año T ---
    # El contexto se toma del lado SIMCE y se normaliza a texto canónico, para que
    # el .replace() de los scripts de modelado sea inofensivo y los datos sean
    # consistentes (las fuentes traen códigos y texto mezclados según el año).
    simce = simce.copy()
    simce["cod_grupo"] = _mapear(simce["cod_grupo"], NSE_ORDINAL).map(NSE_LABEL)
    simce["cod_rural_rbd"] = _mapear(simce["cod_rural_rbd"], RURAL_LABEL,
                                     conservar_original=True)
    simce["cod_depe2"] = _mapear(simce["cod_depe2"], DEPE_LABEL,
                                 conservar_original=True)
    simce_t = simce[KEY + ["prom_mate_rbd", "prom_lect_rbd",
                           "cod_grupo", "cod_rural_rbd", "cod_depe2"]].rename(
        columns={"prom_mate_rbd": "prom_mate2m_rbd", "prom_lect_rbd": "prom_lect2m_rbd"}
    )

    # --- IDPS (indicadores), año T ---
    # Se renombran al esquema histórico ind_* que esperan los modelos.
    idps_t = idps[KEY + ["idps_am", "idps_cc", "idps_hv", "idps_pf"]].rename(
        columns={"idps_am": "ind_am", "idps_cc": "ind_cc",
                 "idps_hv": "ind_hv", "idps_pf": "ind_pf"}
    )

    # --- Join contemporáneo (mismo año), todos los cursos ---
    historico = simce_t.merge(idps_t, on=KEY, how="inner")

    # --- Filtro de validez: descartar puntajes SIMCE inválidos (errores de fuente) ---
    antes = len(historico)
    valido = ((historico["prom_mate2m_rbd"] >= UMBRAL_SIMCE_MIN) &
              (historico["prom_lect2m_rbd"] >= UMBRAL_SIMCE_MIN))
    historico = historico[valido]
    construir_historico.filas_invalidas = antes - len(historico)

    # Orden final de columnas (estilo dataset_historico_final.csv + curso)
    columnas = [
        "rbd", "agno", "curso",
        "prom_mate2m_rbd", "prom_lect2m_rbd",
        "cod_grupo", "cod_rural_rbd", "cod_depe2",
        "ind_am", "ind_cc", "ind_hv", "ind_pf",
    ]
    historico = historico[columnas].sort_values(KEY).reset_index(drop=True)
    return historico


def reportar(historico):
    print(f"Dataset histórico completo: {historico.shape[0]} filas x {historico.shape[1]} columnas")
    invalidas = getattr(construir_historico, "filas_invalidas", None)
    if invalidas is not None:
        print(f"Filas descartadas por puntaje SIMCE inválido (<{UMBRAL_SIMCE_MIN}): {invalidas}")
    print(f"\nRango de puntajes (debe ser >= {UMBRAL_SIMCE_MIN}):")
    print(f"  prom_mate2m_rbd: min={historico['prom_mate2m_rbd'].min()}, max={historico['prom_mate2m_rbd'].max()}")
    print(f"  prom_lect2m_rbd: min={historico['prom_lect2m_rbd'].min()}, max={historico['prom_lect2m_rbd'].max()}")
    print("\nFilas por curso:")
    print(historico["curso"].value_counts().to_string())
    print("\nFilas por año:")
    print(historico["agno"].value_counts().sort_index().to_string())
    print("\nNulos por columna:")
    print(historico.isnull().sum().to_string())
    print("\nValores de contexto (deben ser texto canónico):")
    for col in ["cod_grupo", "cod_rural_rbd", "cod_depe2"]:
        print(f"  {col}: {sorted(historico[col].dropna().unique().tolist())}")
    dup = historico.duplicated(subset=KEY).sum()
    print(f"\nClave (rbd, agno, curso) duplicada: {dup} filas")


def main():
    idps, simce = cargar()
    historico = construir_historico(idps, simce)
    reportar(historico)
    historico.to_csv(OUT_CSV, index=False)
    print(f"\nGuardado en: {OUT_CSV}")


if __name__ == "__main__":
    main()
