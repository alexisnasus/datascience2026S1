#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset_maestro.py — Consolidación final (etapa de join).

Une los dos datasets consolidados (IDPS y SIMCE) en un único dataset maestro
con estructura PREDICTIVA TEMPORAL (desfase de un año):

    Predictores del año T-1   ->   Target SIMCE del año T

Predictores (medidos en T-1):
    idps_am_prev, idps_cc_prev, idps_hv_prev, idps_pf_prev  (IDPS, 0-100)
    simce_mate_prev, simce_lect_prev                        (SIMCE previo, autorregresivo)
Contexto (de T-1):
    nse_ord (ordinal 1-5), curso_ord, cod_rural_rbd, cod_depe2
Targets (año T):
    target_mate, target_lect

Salida: data/processed/dataset_maestro.csv  (una fila por (rbd, agno=T, curso)).

Las rutas se resuelven desde __file__, así que el script corre desde cualquier CWD.
NO imputa, NO escala y NO elimina outliers: esas decisiones (Scrub de modelado)
viven en el notebook/pipeline para evitar fuga de información (data leakage).
"""
from pathlib import Path

import pandas as pd

# --- Rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED = BASE_DIR / "data" / "processed"
IDPS_CSV = PROCESSED / "dataset_consolidado_idps.csv"
SIMCE_CSV = PROCESSED / "dataset_simce_consolidado.csv"
OUT_CSV = PROCESSED / "dataset_maestro.csv"

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
# Etiqueta canónica del NSE a partir del ordinal (columna de referencia legible).
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
# Nivel de escolaridad como variable ordinal (años de escolaridad aprox.).
CURSO_ORD = {"4b": 4, "6b": 6, "8b": 8, "2m": 10}


def _norm_key(valor):
    """Normaliza un valor a una clave comparable: minúsculas y sin espacios."""
    return str(valor).strip().lower()


def _mapear(serie, mapa, *, conservar_original=False):
    """Mapea una serie usando _norm_key. NaN se mantiene como NaN.

    Si conservar_original es True, los valores no reconocidos se dejan tal cual;
    si no, se convierten en NaN (útil para el ordinal estricto del NSE).
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
    df = df[df["curso"].isin(CURSO_ORD)]
    return df


def construir_maestro(idps, simce):
    idps = limpiar_claves(idps)
    simce = limpiar_claves(simce)

    # Evita filas duplicadas por clave (los consolidados ya deberían ser únicos).
    idps = idps.drop_duplicates(subset=KEY)
    simce = simce.drop_duplicates(subset=KEY)

    # --- Target: SIMCE del año T ---
    simce_t = simce[KEY + ["prom_mate_rbd", "prom_lect_rbd"]].rename(
        columns={"prom_mate_rbd": "target_mate", "prom_lect_rbd": "target_lect"}
    )

    # --- Features IDPS medidas en T-1 (se desplaza el año: agno -> agno + 1) ---
    idps_prev = idps[KEY + ["idps_am", "idps_cc", "idps_hv", "idps_pf"]].copy()
    idps_prev["agno"] = idps_prev["agno"] + 1
    idps_prev = idps_prev.rename(columns={
        "idps_am": "idps_am_prev", "idps_cc": "idps_cc_prev",
        "idps_hv": "idps_hv_prev", "idps_pf": "idps_pf_prev",
    })

    # --- SIMCE previo (autorregresivo) + contexto, medidos en T-1 ---
    simce_prev = simce.copy()
    simce_prev["nse_ord"] = _mapear(simce_prev["cod_grupo"], NSE_ORDINAL)
    simce_prev["cod_rural_rbd"] = _mapear(simce_prev["cod_rural_rbd"], RURAL_LABEL,
                                          conservar_original=True)
    simce_prev["cod_depe2"] = _mapear(simce_prev["cod_depe2"], DEPE_LABEL,
                                       conservar_original=True)
    simce_prev = simce_prev[KEY + ["prom_mate_rbd", "prom_lect_rbd",
                                   "nse_ord", "cod_rural_rbd", "cod_depe2"]].copy()
    simce_prev["agno"] = simce_prev["agno"] + 1
    simce_prev = simce_prev.rename(columns={
        "prom_mate_rbd": "simce_mate_prev", "prom_lect_rbd": "simce_lect_prev",
    })

    # --- Join: exige features IDPS(T-1) y SIMCE(T-1) para el target SIMCE(T) ---
    # El inner join garantiza año consecutivo y excluye el salto 2019->2022 (COVID).
    maestro = (
        simce_t
        .merge(idps_prev, on=KEY, how="inner")
        .merge(simce_prev, on=KEY, how="inner")
    )

    # Variables derivadas / canónicas
    maestro["curso_ord"] = maestro["curso"].map(CURSO_ORD)
    maestro["cod_grupo"] = maestro["nse_ord"].map(NSE_LABEL)

    # Garantiza que existan ambos targets (defensivo; SIMCE ya hizo dropna).
    maestro = maestro.dropna(subset=["target_mate", "target_lect"])

    # Orden final de columnas (agno = año-objetivo T)
    columnas = [
        "rbd", "agno", "curso", "curso_ord", "nse_ord", "cod_grupo",
        "cod_rural_rbd", "cod_depe2",
        "idps_am_prev", "idps_cc_prev", "idps_hv_prev", "idps_pf_prev",
        "simce_mate_prev", "simce_lect_prev",
        "target_mate", "target_lect",
    ]
    maestro = maestro[columnas].sort_values(KEY).reset_index(drop=True)
    return maestro


def reportar(maestro):
    print(f"Dataset maestro: {maestro.shape[0]} filas x {maestro.shape[1]} columnas")
    print("\nFilas por año-objetivo (T):")
    print(maestro["agno"].value_counts().sort_index().to_string())
    print("\nFilas por curso:")
    print(maestro["curso"].value_counts().to_string())
    print("\nNulos por columna:")
    print(maestro.isnull().sum().to_string())
    dup = maestro.duplicated(subset=KEY).sum()
    print(f"\nClave (rbd, agno, curso) duplicada: {dup} filas")


def main():
    idps, simce = cargar()
    maestro = construir_maestro(idps, simce)
    reportar(maestro)
    maestro.to_csv(OUT_CSV, index=False)
    print(f"\nGuardado en: {OUT_CSV}")


if __name__ == "__main__":
    main()
