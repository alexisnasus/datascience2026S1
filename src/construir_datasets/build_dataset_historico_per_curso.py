#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset_historico_per_curso.py — Join CONTEMPORÁNEO separado POR CURSO.

Variante per-curso de build_dataset_historico.py. En vez de un único dataset con
los 4 cursos mezclados, genera UN archivo por nivel educativo (4b, 6b, 8b, 2m):

    data/processed/por_curso/dataset_historico_{curso}.csv

Cada archivo tiene el MISMO esquema que dataset_historico_completo.csv, pero una
sola clave de curso por archivo. Es la entrada de datascienceproyecto1_per_curso.py,
que estudia el efecto de los IDPS curso por curso (con recorte de outliers IQR
calculado por curso, sobre filas homogéneas, en vez de global).

Predictores y target del MISMO año T (análisis descriptivo-explicativo), igual que
build_dataset_historico.py. Las filas resultantes son idénticas a dividir por curso
el dataset completo; aquí se construyen desde los consolidados para dejar la pista
per-curso autónoma.

Reusa las utilidades de normalización de build_dataset_historico.py importándolas
(es import-seguro: ese módulo protege su ejecución con if __name__ == "__main__").

Las rutas se resuelven desde __file__, así que el script corre desde cualquier CWD.
NO imputa, NO escala y NO elimina outliers/nulos: esas decisiones (Scrub de
modelado) viven en el script de modelado.
"""
from pathlib import Path

# Reusa la lógica de normalización/limpieza del join completo (import-seguro).
from build_dataset_historico import (
    cargar,
    limpiar_claves,
    _mapear,
    NSE_ORDINAL,
    NSE_LABEL,
    RURAL_LABEL,
    DEPE_LABEL,
    KEY,
    UMBRAL_SIMCE_MIN,
)

# --- Rutas ---
BASE_DIR = Path(__file__).resolve().parents[2]  # src/construir_datasets/<archivo>.py -> raíz
PROCESSED = BASE_DIR / "data" / "processed"
POR_CURSO_DIR = PROCESSED / "por_curso"

# Orden de progresión educativa (básica -> media). Es también el orden en que se
# reportan y, más adelante, el eje X del gráfico de evolución del script de modelado.
CURSOS = ["4b", "6b", "8b", "2m"]

# Esquema final por archivo (idéntico a dataset_historico_completo.csv).
COLUMNAS = [
    "rbd", "agno", "curso",
    "prom_mate2m_rbd", "prom_lect2m_rbd",
    "cod_grupo", "cod_rural_rbd", "cod_depe2",
    "ind_am", "ind_cc", "ind_hv", "ind_pf",
]


def preparar_fuentes():
    """Carga y homologa claves de ambos consolidados una sola vez."""
    idps, simce = cargar()
    idps = limpiar_claves(idps).drop_duplicates(subset=KEY)
    simce = limpiar_claves(simce).drop_duplicates(subset=KEY)
    return idps, simce


def construir_curso(idps, simce, curso):
    """Join contemporáneo IDPS<->SIMCE para un único curso, ya normalizado y filtrado."""
    idps_c = idps[idps["curso"] == curso].copy()
    simce_c = simce[simce["curso"] == curso].copy()

    # --- SIMCE (puntajes + contexto), normalizado a texto canónico ---
    simce_c["cod_grupo"] = _mapear(simce_c["cod_grupo"], NSE_ORDINAL).map(NSE_LABEL)
    simce_c["cod_rural_rbd"] = _mapear(simce_c["cod_rural_rbd"], RURAL_LABEL,
                                       conservar_original=True)
    simce_c["cod_depe2"] = _mapear(simce_c["cod_depe2"], DEPE_LABEL,
                                   conservar_original=True)
    simce_t = simce_c[KEY + ["prom_mate_rbd", "prom_lect_rbd",
                             "cod_grupo", "cod_rural_rbd", "cod_depe2"]].rename(
        columns={"prom_mate_rbd": "prom_mate2m_rbd", "prom_lect_rbd": "prom_lect2m_rbd"}
    )

    # --- IDPS (indicadores), renombrados al esquema histórico ind_* ---
    idps_t = idps_c[KEY + ["idps_am", "idps_cc", "idps_hv", "idps_pf"]].rename(
        columns={"idps_am": "ind_am", "idps_cc": "ind_cc",
                 "idps_hv": "ind_hv", "idps_pf": "ind_pf"}
    )

    # --- Join contemporáneo (mismo año) ---
    historico = simce_t.merge(idps_t, on=KEY, how="inner")

    # --- Filtro de validez: descartar puntajes SIMCE inválidos (errores de fuente) ---
    antes = len(historico)
    valido = ((historico["prom_mate2m_rbd"] >= UMBRAL_SIMCE_MIN) &
              (historico["prom_lect2m_rbd"] >= UMBRAL_SIMCE_MIN))
    historico = historico[valido]
    invalidas = antes - len(historico)

    historico = historico[COLUMNAS].sort_values(KEY).reset_index(drop=True)
    return historico, invalidas


def reportar_curso(curso, historico, invalidas, ruta):
    print(f"\n=== Curso {curso} ===")
    print(f"  Filas: {historico.shape[0]} x {historico.shape[1]} columnas")
    print(f"  Filas descartadas por puntaje SIMCE inválido (<{UMBRAL_SIMCE_MIN}): {invalidas}")
    print(f"  Años: {sorted(historico['agno'].unique())}")
    print(f"  Rango puntajes (debe ser >= {UMBRAL_SIMCE_MIN}):")
    print(f"    prom_mate2m_rbd: min={historico['prom_mate2m_rbd'].min()}, "
          f"max={historico['prom_mate2m_rbd'].max()}")
    print(f"    prom_lect2m_rbd: min={historico['prom_lect2m_rbd'].min()}, "
          f"max={historico['prom_lect2m_rbd'].max()}")
    print(f"  Nulos por columna:")
    nulos = historico.isnull().sum()
    print("    " + nulos[nulos > 0].to_string().replace("\n", "\n    ")
          if nulos.sum() else "    (sin nulos)")
    print(f"  Guardado en: {ruta}")


def main():
    POR_CURSO_DIR.mkdir(parents=True, exist_ok=True)
    idps, simce = preparar_fuentes()

    total = 0
    for curso in CURSOS:
        historico, invalidas = construir_curso(idps, simce, curso)
        ruta = POR_CURSO_DIR / f"dataset_historico_{curso}.csv"
        historico.to_csv(ruta, index=False)
        reportar_curso(curso, historico, invalidas, ruta)
        total += len(historico)

    print(f"\nTotal de filas en los {len(CURSOS)} archivos por curso: {total}")


if __name__ == "__main__":
    main()
