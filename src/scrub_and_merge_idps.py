import pandas as pd
import glob
import os
import re

# Configuración de rutas dinámicas
BASE_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(BASE_PROJECT_DIR, "data", "raw", "idps")

# Asegurar la creación del directorio de salida
os.makedirs(os.path.join(BASE_PROJECT_DIR, "data", "processed"), exist_ok=True)
OUTPUT_FILE = os.path.join(BASE_PROJECT_DIR, "data", "processed", "dataset_consolidado_idps.csv")

# Diccionario de estandarización de columnas extendido (incluyendo dimensiones y variables descriptivas)
COLUMN_MAPPING = {
    'rbd': 'rbd', 'id_rbd': 'rbd',
    'agno': 'agno', 'ano': 'agno',
    'grado': 'curso',
    
    # IDPS Generales
    'ind_am': 'idps_am', 'ind_am_rbd': 'idps_am',
    'ind_cc': 'idps_cc', 'ind_cc_rbd': 'idps_cc',
    'ind_hv': 'idps_hv', 'ind_hv_rbd': 'idps_hv',
    'ind_pf': 'idps_pf', 'ind_pf_rbd': 'idps_pf',
    
    # Sub-dimensiones IDPS (Ejemplo: 2018 y posteriores)
    'dim_am_aa_rbd': 'dim_am_aa', 'dim_am_me_rbd': 'dim_am_me',
    'dim_cc_ao_rbd': 'dim_cc_ao', 'dim_cc_ar_rbd': 'dim_cc_ar', 'dim_cc_as_rbd': 'dim_cc_as',
    'dim_hv_ac_rbd': 'dim_hv_ac', 'dim_hv_ha_rbd': 'dim_hv_ha', 'dim_hv_va_rbd': 'dim_hv_va',
    'dim_pf_pa_rbd': 'dim_pf_pa', 'dim_pf_sp_rbd': 'dim_pf_sp', 'dim_pf_vd_rbd': 'dim_pf_vd',
    
    # Socioeconómicas y geográficas
    'cod_depe2': 'cod_depe2',
    'cod_grupo': 'cod_grupo',
    'cod_rural_rbd': 'cod_rural_rbd',
    'nom_rbd': 'nom_rbd',
    'cod_reg_rbd': 'cod_reg_rbd',
    'cod_pro_rbd': 'cod_pro_rbd',
    'cod_com_rbd': 'cod_com_rbd',
    'codigo_bbdd': 'codigo_bbdd', 
    'fecha_bbdd': 'fecha_bbdd',
    
    # SIMCE
    'prom_mate': 'simce_mate',
    'prom_lect': 'simce_lect'
}

def clean_column_names(df):
    """Limpia y estandariza los nombres de columnas de un DataFrame."""
    df.columns = df.columns.str.lower().str.strip()
    df.rename(columns=COLUMN_MAPPING, inplace=True)
    return df

def infer_separator(filepath):
    """Infiere si el delimitador es coma, punto y coma o pipe (|)."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
        first_line = file.readline()
        if '|' in first_line: return '|'
        if ';' in first_line: return ';'
        return ',' # Por defecto, la coma (para 2016 y txt general)

def handle_long_format(df):
    """Pivotea df si viene en formato long permitiendo índices y sub-dimensiones."""
    if 'ind' in df.columns and 'prom' in df.columns and 'rbd' in df.columns:
        idx_cols = [c for c in df.columns if c not in ['ind', 'prom']]

        # Estandarizar valores de la columna 'ind'
        df['ind'] = df['ind'].str.lower().str.strip()

        # Mapeo manual para los índices principales dentro de la columna ind
        val_mapping = {
            'am': 'idps_am', 'aa': 'idps_am',  # En base 2023 AA es Autoestima Académica
            'cc': 'idps_cc',
            'hv': 'idps_hv',
            'pf': 'idps_pf'
        }
        df['ind'] = df['ind'].map(lambda x: val_mapping.get(x, f'dim_{x}' if not x.startswith('idps_') and not x.startswith('dim_') else x))

        # Pivotear: el index agrupa, columns son los indicadores, values es el puntaje
        df = df.pivot_table(index=idx_cols, columns='ind', values='prom', aggfunc='max').reset_index()
        # Limpiar nombre de los márgenes del pivot
        df.columns.name = None
    return df

# Agrupador temporal por curso y año
dfs_by_period = {}

print("Iniciando ETL de Datos...")
for subdir in os.listdir(BASE_DIR):
    subdir_path = os.path.join(BASE_DIR, subdir)
    
    if os.path.isdir(subdir_path):
        curso = subdir.lower() # Ejemplo: '2m', '4b'
        
        file_pattern = os.path.join(subdir_path, '*.*')
        for filepath in glob.glob(file_pattern):
            if filepath.endswith('.csv') or filepath.endswith('.txt'):
                filename = os.path.basename(filepath)
                
                try:
                    # Detectar separador
                    sep = infer_separator(filepath)
                    
                    df_temp = pd.read_csv(filepath, sep=sep, encoding='latin1', low_memory=False)
                    
                    # Limpieza estándar
                    df_temp = clean_column_names(df_temp)
                    
                    # Asegurar la existencia de "curso"
                    if 'curso' not in df_temp.columns:
                        df_temp['curso'] = curso
                        
                    # Extraer AGNO del nombre del archivo si no viene explícito
                    match = re.search(r'(20\d{2})', filename)
                    agno_file = int(match.group(1)) if match else None
                    
                    if 'agno' not in df_temp.columns and agno_file:
                        df_temp['agno'] = agno_file
                        
                    # El año real mandatorio para la llave de cruce
                    current_agno = df_temp['agno'].iloc[0] if 'agno' in df_temp.columns and pd.notna(df_temp['agno'].iloc[0]) else agno_file
                    
                    # Manejar pivot en caso de 2023+ (Formato long)
                    df_temp = handle_long_format(df_temp)
                    
                    # Guardar en el diccionario agrupador
                    key = (curso, current_agno)
                    if key not in dfs_by_period:
                        dfs_by_period[key] = []
                    dfs_by_period[key].append(df_temp)
                    
                    print(f"[OK] Leído: {filename} -> Agrupado en {key}")
                    
                except Exception as e:
                    print(f"[ERROR] No se pudo leer {filename}: {e}")

# 1. Merge Horizontal (por año y curso)
master_list = []
for key, list_dfs in dfs_by_period.items():
    # Merge múltiple para todos los dataframes que comparten (curso, agno)
    df_merged = list_dfs[0]
    for df_next in list_dfs[1:]:
        # Buscamos las columnas que tienen en común para que no generen "rbd_x", "rbd_y"
        common_cols = list(set(df_merged.columns) & set(df_next.columns))
        df_merged = pd.merge(df_merged, df_next, on=common_cols, how='outer')
        
    master_list.append(df_merged)

# 2. Append Vertical (toda la historia)
if master_list:
    df_final = pd.concat(master_list, ignore_index=True)
    
    # Quitar duplicados por establecimiento, año y curso asegurándonos de conservar la fila
    df_final.drop_duplicates(subset=['rbd', 'agno', 'curso'], keep='last', inplace=True)
    
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ ETL Terminado con Éxito: Dataset maestro guardado en '{OUTPUT_FILE}' de dimensiones {df_final.shape}")
else:
    print("Ningún archivo fue leído con éxito.")