import os
import re
import pandas as pd
from pathlib import Path

# Configuraciones
GLOSA_SIMCE_DIR = Path("data/glosa_simce")
GLOSA_IDPS_DIR = Path("data/glosas_idps")
OUTPUT_FILE = Path("data/processed/todas_las_glosas.csv")

def extract_metadata(filename, origin):
    """Extrae origen, curso y año del nombre del archivo."""
    curso = "general"
    agno = "desconocido"
    
    # Patrón común: simce2m2016, idps4b2023, idps2016
    match = re.search(r'(simce|idps)([2468][mb])?(\d{4})', filename, re.IGNORECASE)
    if match:
        if match.group(2):
            curso = match.group(2).lower()
        agno = match.group(3)
    else:
        # Fallback solo año
        year_match = re.search(r'(\d{4})', filename)
        if year_match:
            agno = year_match.group(1)
            
    return origin, curso, agno

def normalize_columns(columns):
    """Busca mapear las columnas a 'variable' y 'descripcion' independientemente del año."""
    norm_cols = []
    for c in columns:
        c_str = str(c).lower().strip()
        # mapeo de variaciones comunes
        if any(x in c_str for x in ['nombre', 'variable', 'código']):
            norm_cols.append('variable')
        elif any(x in c_str for x in ['descrip', 'etiqueta', 'label']):
            norm_cols.append('descripcion')
        else:
            norm_cols.append(c_str)
    return norm_cols

def process_files():
    all_rows = []
    
    directories = [
        (GLOSA_SIMCE_DIR, "simce"),
        (GLOSA_IDPS_DIR, "idps")
    ]
    
    for dir_path, origin in directories:
        if not dir_path.exists():
            print(f"Directorio no encontrado: {dir_path}")
            continue
            
        for file in dir_path.glob("*.xlsx"):
            if file.name.startswith("~$"):
                continue
                
            origin_val, curso_val, agno_val = extract_metadata(file.name, origin)
            
            try:
                # Leer todas las hojas
                xls = pd.read_excel(file, sheet_name=None)
                
                for sheet_name, df in xls.items():
                    if 'ndice' in sheet_name.lower():
                        continue
                        
                    # Buscar la fila que tiene 'Nombre' o 'Variable' para setear como header
                    # usualmente está alrededor de la fila 7 o 6 (índice 6 o 7 si incluye el header de pandas por defecto)
                    # Convertiremos el df a headerless temporalmente para buscar
                    header_idx = -1
                    
                    # Chequear en los nombres de columnas primero por si acaso
                    cols_str = [str(c).lower().strip() for c in df.columns]
                    if any('nombre' in c or 'variable' in c for c in cols_str):
                        # El header ya está en la columna
                        df_var = df.copy()
                    else:
                        for idx, row in df.iterrows():
                            # Revisar las primeras columnas por un match más exacto
                            row_vals = [str(x).lower().strip() for x in row.values[:3]]
                            if any(v in ['nombre', 'variable', 'nombre variable', 'nombre de la variable', 'nombre de variable'] for v in row_vals):
                                header_idx = idx
                                break
                                
                        if header_idx != -1:
                            # Seteamos esa fila como header
                            new_header = df.iloc[header_idx]
                            df_var = df.iloc[header_idx+1:].copy()
                            df_var.columns = new_header
                        else:
                            continue # No encontramos la tabla de variables

                    if len(df_var.columns) < 2:
                        continue
                        
                    original_cols = list(df_var.columns)
                    df_var.columns = normalize_columns(df_var.columns)
                    
                    if 'variable' in df_var.columns and 'descripcion' in df_var.columns:
                        df_var = df_var.loc[:, ~df_var.columns.duplicated()]
                        df_var = df_var[['variable', 'descripcion']].copy()
                    else:
                        df_var = df_var.iloc[:, :2].copy()
                        df_var.columns = ['variable', 'descripcion']
                        
                    # Detener cuando hay NaN en variable (fin de la tabla o inicio de valores categóricos)
                    # df_var = df_var.dropna(subset=['variable']) # en vez de borrar todo, cortamos en el primer NaN
                    
                    valid_rows = []
                    for _, row in df_var.iterrows():
                        if pd.isna(row['variable']) or str(row['variable']).strip() == '' or str(row['variable']).lower() == 'nan':
                            break # Corta en la primera línea vacía
                        valid_rows.append(row)
                        
                    df_var = pd.DataFrame(valid_rows)
                    if df_var.empty:
                        continue
                        
                    # Actualizar curso_val si la hoja indica curso (ej. idps2m vs idps4b)
                    current_curso = curso_val
                    if "2m" in sheet_name.lower(): current_curso = "2m"
                    elif "4b" in sheet_name.lower(): current_curso = "4b"
                    elif "6b" in sheet_name.lower(): current_curso = "6b"
                    elif "8b" in sheet_name.lower(): current_curso = "8b"
                    
                    df_var['contexto'] = f"{origin_val}_{current_curso}_{agno_val}_{sheet_name}"
                    
                    all_rows.append(df_var)
            except Exception as e:
                print(f"Error procesando {file.name}: {e}")
                
    if not all_rows:
        print("No se encontraron datos.")
        return
        
    final_df = pd.concat(all_rows, ignore_index=True)
    
    # Limpieza: sacar espacios y convertir a string
    final_df['variable'] = final_df['variable'].astype(str).str.strip().str.lower()
    # Eliminar vacíos
    final_df = final_df[final_df['variable'] != 'nan']
    final_df = final_df[final_df['variable'] != '']
    
    print(f"Se cargaron un total de {len(final_df)} variables (con duplicados). Consolidando...")
    
    # Consolidación: Deduplicar agregando los contextos en una sola columna string
    def combine_contexts(series):
        # separar contextos, deduplicar y sortear
        return ", ".join(sorted(list(set(series))))
        
    def first_valid_description(series):
        # Retorna la primera descripción no nula
        valid = series.dropna()
        valid = valid[valid != '']
        return next(iter(valid), "Sin descripción")
        
    grouped = final_df.groupby('variable').agg({
        'descripcion': first_valid_description, 
        'contexto': combine_contexts
    }).reset_index()
    
    # Renombrar para claridad
    grouped.rename(columns={'contexto': 'archivos_presentes'}, inplace=True)
    
    # Guardar version consolidada
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # Separar en SIMCE e IDPS
    # Una variable puede estar en ambos (ej: 'agno', 'rbd'), así que usamos str.contains
    df_simce = grouped[grouped['archivos_presentes'].str.contains('simce', case=False, na=False)].copy()
    df_idps = grouped[grouped['archivos_presentes'].str.contains('idps', case=False, na=False)].copy()
    
    output_simce = OUTPUT_FILE.parent / "todas_las_glosas_simce.csv"
    output_idps = OUTPUT_FILE.parent / "todas_las_glosas_idps.csv"
    
    df_simce.to_csv(output_simce, index=False, encoding='utf-8-sig')
    df_idps.to_csv(output_idps, index=False, encoding='utf-8-sig')
    
    print(f"¡Éxito! Se procesaron {len(all_rows)} diccionarios.")
    print(f"Se extrajeron {len(grouped)} variables únicas consolidadas ({OUTPUT_FILE.name}).")
    print(f" -> {len(df_simce)} variables de SIMCE ({output_simce.name}).")
    print(f" -> {len(df_idps)} variables de IDPS ({output_idps.name}).")

if __name__ == "__main__":
    process_files()
