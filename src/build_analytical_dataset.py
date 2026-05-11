"""Construye dataset_historico_final.csv con el schema que espera DataScienceProyecto1.ipynb.

Filtra a II Medio (curso='2m'), cruza IDPS x SIMCE por (rbd, agno) y emite las
11 columnas que el notebook lee: rbd, agno, prom_mate2m_rbd, prom_lect2m_rbd,
cod_grupo, cod_rural_rbd, cod_depe2, ind_am, ind_cc, ind_hv, ind_pf.
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
SRC = os.path.join(BASE_DIR, "src")

IDPS_PATH = os.path.join(PROCESSED, "dataset_consolidado_idps.csv")
SIMCE_PATH = os.path.join(PROCESSED, "dataset_simce_consolidado.csv")

# Dos destinos: la ubicación canónica (data/processed) y una copia junto al
# notebook (src/), porque el notebook hace pd.read_csv("dataset_historico_final.csv")
# con ruta relativa y el usuario no puede editarlo.
OUTPUT_PATHS = [
    os.path.join(PROCESSED, "dataset_historico_final.csv"),
    os.path.join(SRC, "dataset_historico_final.csv"),
]

IDPS_COLS = ['rbd', 'agno', 'curso', 'idps_am', 'idps_cc', 'idps_hv', 'idps_pf']
IDPS_RENAME = {'idps_am': 'ind_am', 'idps_cc': 'ind_cc',
               'idps_hv': 'ind_hv', 'idps_pf': 'ind_pf'}

SIMCE_COLS = ['rbd', 'agno', 'curso',
              'prom_mate_rbd', 'prom_lect_rbd',
              'cod_grupo', 'cod_depe2', 'cod_rural_rbd']
SIMCE_RENAME = {'prom_mate_rbd': 'prom_mate2m_rbd',
                'prom_lect_rbd': 'prom_lect2m_rbd'}

FINAL_COLUMNS = ['rbd', 'agno', 'prom_mate2m_rbd', 'prom_lect2m_rbd',
                 'cod_grupo', 'cod_rural_rbd', 'cod_depe2',
                 'ind_am', 'ind_cc', 'ind_hv', 'ind_pf']

# El IDPS de 2022/2023 trae curso truncado ('2' en vez de '2m', '4' en vez de '4b').
CURSO_NORMALIZE = {'2': '2m', '4': '4b', '6': '6b', '8': '8b'}


def main():
    idps = pd.read_csv(IDPS_PATH, usecols=IDPS_COLS, encoding='utf-8-sig')
    simce = pd.read_csv(SIMCE_PATH, usecols=SIMCE_COLS, encoding='utf-8-sig')

    for df_ in (idps, simce):
        df_['rbd'] = pd.to_numeric(df_['rbd'], errors='coerce').astype('Int64')
        df_['agno'] = pd.to_numeric(df_['agno'], errors='coerce').astype('Int64')
        df_['curso'] = df_['curso'].astype(str).str.lower().str.strip()
        df_['curso'] = df_['curso'].replace(CURSO_NORMALIZE)

    idps = idps[idps['curso'] == '2m'].drop(columns=['curso'])
    simce = simce[simce['curso'] == '2m'].drop(columns=['curso'])

    idps = idps.rename(columns=IDPS_RENAME)
    simce = simce.rename(columns=SIMCE_RENAME)

    # Consolidar IDPS: una fila por (rbd, agno). mean() con skipna=True
    # absorbe el caso del formato Long no pivoteado de 2023+, donde cada
    # (rbd, agno) tiene varias filas con un solo indicador no nulo.
    idps = (idps.groupby(['rbd', 'agno'], as_index=False)
                 [['ind_am', 'ind_cc', 'ind_hv', 'ind_pf']]
                 .mean())

    # Consolidar SIMCE por seguridad (no-op si fusion_simce.py ya dedupea).
    simce = (simce.groupby(['rbd', 'agno'], as_index=False)
                  .agg({'prom_mate2m_rbd': 'mean',
                        'prom_lect2m_rbd': 'mean',
                        'cod_grupo': 'first',
                        'cod_depe2': 'first',
                        'cod_rural_rbd': 'first'}))

    merged = simce.merge(idps, on=['rbd', 'agno'], how='inner')
    merged = merged[FINAL_COLUMNS]

    for path in OUTPUT_PATHS:
        merged.to_csv(path, index=False, encoding='utf-8')
        print(f"Escrito: {path}")

    print(f"\nFilas: {len(merged)} | Columnas: {merged.shape[1]}")
    print(f"Anos: {sorted(merged['agno'].dropna().unique().tolist())}")
    completas = merged[['ind_am', 'ind_cc', 'ind_hv', 'ind_pf']].dropna().shape[0]
    print(f"Filas con los 4 indices IDPS completos: {completas}")


if __name__ == "__main__":
    main()
