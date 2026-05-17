"""Fase OBTAIN: consolida TODOS los IDPS (2016-2025, todos los cursos) en un
unico CSV wide y limpio.

Esto NO es Scrub: no se filtra por curso/ano, no se hace dropna, no se imputa
ni se armonizan los codigos vs etiquetas (2017 trae 'Municipal'/'Bajo' como
texto; 2018+ trae codigos numericos). Esas decisiones se toman despues, en la
fase Scrub, sobre este consolidado.

Salida: data/processed/dataset_consolidado_idps.csv
Una fila por (rbd, agno, curso) con los 4 indices IDPS + contexto.
"""
import pandas as pd
import glob
import os
import re

# Configuracion de rutas dinamicas
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fuente canonica versionada en el repo (no se usa data/raw/, que era una
# copia manual redundante).
BASE_DIR = os.path.join(BASE_PROJECT_DIR, "data", "agrupado", "idps")
os.makedirs(os.path.join(BASE_PROJECT_DIR, "data", "processed"), exist_ok=True)
OUTPUT_FILE = os.path.join(BASE_PROJECT_DIR, "data", "processed",
                           "dataset_consolidado_idps.csv")

# Mapea variantes de nombre (incluye el caso MAYUSCULAS/alterno de 2018) al
# schema canonico. Lo que no este aqui ni en OUTPUT_COLUMNS se descarta.
COLUMN_MAPPING = {
    'rbd': 'rbd', 'id_rbd': 'rbd',
    'agno': 'agno', 'ano': 'agno',
    # Indicadores wide (2016-2019): 'ind_am' o 'ind_am_rbd'
    'ind_am': 'idps_am', 'ind_am_rbd': 'idps_am',
    'ind_cc': 'idps_cc', 'ind_cc_rbd': 'idps_cc',
    'ind_hv': 'idps_hv', 'ind_hv_rbd': 'idps_hv',
    'ind_pf': 'idps_pf', 'ind_pf_rbd': 'idps_pf',
    # Contexto geografico / socioeconomico (nombre canonico)
    'nom_rbd': 'nom_rbd',
    'cod_reg_rbd': 'cod_reg_rbd', 'nom_reg_rbd': 'nom_reg_rbd',
    'cod_pro_rbd': 'cod_pro_rbd', 'nom_pro_rbd': 'nom_pro_rbd',
    'cod_com_rbd': 'cod_com_rbd', 'nom_com_rbd': 'nom_com_rbd',
    'nom_deprov_rbd': 'nom_deprov_rbd',
    'cod_depe2': 'cod_depe2', 'cod_grupo': 'cod_grupo',
    'cod_rural_rbd': 'cod_rural_rbd',
    'codigo_bbdd': 'codigo_bbdd', 'codigo_bdd': 'codigo_bbdd',
    'fecha_bbdd': 'fecha_bbdd',
    # Variantes alternas de 2018 (ya en minuscula tras el lower())
    'nom_regi_n': 'nom_reg_rbd',     # "NOM_REGION" -> nom_reg_rbd
    'nom_provincia': 'nom_pro_rbd',
    'nom_comuna': 'nom_com_rbd',
    'nom_deprov': 'nom_deprov_rbd',
    # 'cod_deprov' no tiene equivalente canonico -> se descarta por whitelist
}

# Whitelist final: solo estas columnas salen al consolidado, en este orden.
OUTPUT_COLUMNS = [
    'rbd', 'agno', 'curso',
    'idps_am', 'idps_cc', 'idps_hv', 'idps_pf',
    'nom_rbd', 'cod_reg_rbd', 'nom_reg_rbd',
    'cod_pro_rbd', 'nom_pro_rbd', 'cod_com_rbd', 'nom_com_rbd',
    'nom_deprov_rbd', 'cod_depe2', 'cod_grupo', 'cod_rural_rbd',
    'codigo_bbdd', 'fecha_bbdd',
]

# 'grado' viene truncado/inconsistente ('2', '2m'); el curso se toma SIEMPRE
# del nombre del subdirectorio, que es la fuente confiable.
CURSO_NORMALIZE = {'2': '2m', '4': '4b', '6': '6b', '8': '8b',
                   '2m': '2m', '4b': '4b', '6b': '6b', '8b': '8b'}

# Formato long 2022-2024: columna 'ind' con texto AM/CC/HV/PF.
IND_MAP = {'am': 'idps_am', 'aa': 'idps_am',  # En 2023 'AA' = Autoestima Acad.
           'cc': 'idps_cc', 'hv': 'idps_hv', 'pf': 'idps_pf'}

# Formato long 2025: columna 'id_indicador' con codigos numericos.
# 1=AM (verificado: el archivo _dim_ muestra que el indicador 1 tiene 2
# subdimensiones, y AM es el unico con 2 subdimensiones -> AA, ME),
# luego 2=CC, 3=HV, 4=PF (orden canonico Agencia de Calidad).
ID_INDICADOR_MAP = {'1': 'idps_am', '2': 'idps_cc',
                    '3': 'idps_hv', '4': 'idps_pf'}

KEYS = ['rbd', 'agno', 'curso']


def infer_separator(filepath):
    """Infiere el delimitador (coma / punto y coma / pipe)."""
    with open(filepath, 'r', encoding='latin1', errors='ignore') as f:
        first_line = f.readline()
    if '|' in first_line:
        return '|'
    if ';' in first_line:
        return ';'
    return ','


def is_main_file(filename):
    """True solo para el archivo indice principal por (curso, ano).

    Excluye los archivos de granularidad extra (_dim_, niveles, subdim) que
    contaminaban el consolidado con decenas de columnas.
    """
    low = filename.lower()
    if not (low.endswith('.csv') or low.endswith('.txt')):
        return False
    return not any(tag in low for tag in ('_dim', 'niveles', 'subdim'))


def standardize_columns(df):
    """Minusculas + rename canonico + colapsa duplicados tras el rename."""
    df.columns = df.columns.str.lower().str.strip()
    df = df.rename(columns=COLUMN_MAPPING)
    # Tras el rename pueden quedar nombres repetidos (p.ej. dos -> nom_reg_rbd):
    # conservar la primera aparicion.
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def pivot_long_format(df):
    """Formato long 2022+ -> wide idps_am..idps_pf.

    El indicador viene en la columna 'ind' (texto AM/CC/HV/PF, 2022-2024) o
    'id_indicador' (codigo 1-4, 2025). Pivotea solo sobre las claves estables
    (rbd, agno, curso) para no perder filas por NaN en columnas de contexto,
    y re-une el contexto aparte.
    """
    if 'prom' not in df.columns:
        return df  # Ya viene wide (2016-2019)
    if 'ind' in df.columns:
        ind_col, ind_map = 'ind', IND_MAP
    elif 'id_indicador' in df.columns:
        ind_col, ind_map = 'id_indicador', ID_INDICADOR_MAP
    else:
        return df  # Wide sin columna indicadora

    df = df.copy()
    df['_ind'] = (df[ind_col].astype(str).str.lower().str.strip()
                  .map(ind_map))
    df = df.dropna(subset=['_ind'])
    df['prom'] = pd.to_numeric(df['prom'], errors='coerce')

    keys = [k for k in KEYS if k in df.columns]
    indicadores = (df.pivot_table(index=keys, columns='_ind', values='prom',
                                  aggfunc='mean')
                     .reset_index())
    indicadores.columns.name = None

    # Contexto: un valor por clave (primer no nulo). Solo columnas whitelisteadas.
    ctx_cols = [c for c in df.columns
                if c in OUTPUT_COLUMNS and c not in keys
                and c not in ('idps_am', 'idps_cc', 'idps_hv', 'idps_pf')]
    if ctx_cols:
        contexto = df.groupby(keys, as_index=False)[ctx_cols].first()
        indicadores = indicadores.merge(contexto, on=keys, how='left')
    return indicadores


def main():
    frames = []
    print("Iniciando OBTAIN (consolidacion IDPS)...")

    for subdir in sorted(os.listdir(BASE_DIR)):
        subdir_path = os.path.join(BASE_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue
        curso = CURSO_NORMALIZE.get(subdir.lower(), subdir.lower())

        for filepath in sorted(glob.glob(os.path.join(subdir_path, '*.*'))):
            filename = os.path.basename(filepath)
            if not is_main_file(filename):
                continue
            try:
                sep = infer_separator(filepath)
                df = pd.read_csv(filepath, sep=sep, encoding='latin1',
                                 low_memory=False)
                df = standardize_columns(df)

                # curso desde el subdirectorio (fuente confiable)
                df['curso'] = curso
                # agno desde el nombre del archivo (siempre presente y fiable)
                m = re.search(r'(20\d{2})', filename)
                if m:
                    df['agno'] = int(m.group(1))

                df = pivot_long_format(df)

                # Indices a numerico (estandarizacion de tipo, no es Scrub)
                for col in ('idps_am', 'idps_cc', 'idps_hv', 'idps_pf'):
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                frames.append(df)
                print(f"[OK] {filename} -> ({curso}, {df.shape[0]} filas)")
            except Exception as e:
                print(f"[ERROR] {filename}: {e}")

    if not frames:
        print("Ningun archivo fue leido. Verifica que data/raw/idps este "
              "poblado desde data/agrupado/idps.")
        return

    consolidado = pd.concat(frames, ignore_index=True)

    # Asegurar todas las columnas de la whitelist y el orden fijo
    for col in OUTPUT_COLUMNS:
        if col not in consolidado.columns:
            consolidado[col] = pd.NA
    consolidado = consolidado[OUTPUT_COLUMNS]

    # Una fila por establecimiento/ano/curso (sin filtrar: solo dedup tecnico)
    consolidado = consolidado.drop_duplicates(subset=KEYS, keep='last')

    consolidado.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\nOK Consolidado guardado en {OUTPUT_FILE}")
    print(f"Dimensiones: {consolidado.shape}")
    print(f"Cursos: {sorted(consolidado['curso'].dropna().unique())}")
    print(f"Anos:   {sorted(consolidado['agno'].dropna().astype(int).unique())}")
    completas = consolidado[['idps_am', 'idps_cc', 'idps_hv',
                             'idps_pf']].dropna().shape[0]
    print(f"Filas con los 4 indices IDPS: {completas} / {len(consolidado)}")


if __name__ == "__main__":
    main()
