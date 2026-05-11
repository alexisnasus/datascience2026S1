import os
import glob
import pandas as pd
import re

def consolidar_datos_simce(input_dir: str, output_file: str):
    """
    Consolida, estandariza y limpia archivos SIMCE a nivel de establecimiento (RBD).
    """
    # Buscar todos los archivos que contienen '_rbd' en nombre (archivos a nivel de colegio)
    archivos_rbd = glob.glob(os.path.join(input_dir, "**", "*_rbd_*.csv"), recursive=True) + \
                   glob.glob(os.path.join(input_dir, "**", "*_rbd_*.txt"), recursive=True) + \
                   glob.glob(os.path.join(input_dir, "**", "*_rbd.csv"), recursive=True)
                   
    if not archivos_rbd:
        print("No se encontraron archivos _rbd en el directorio especificado.")
        return

    df_list = []
    
    # Expresión regular para detectar columnas como 'prom_mate2m_rbd', 'prom_lect4b_rbd'
    regex_mate = re.compile(r'^prom_mate\w+_rbd$', re.IGNORECASE)
    regex_lect = re.compile(r'^prom_lect\w+_rbd$', re.IGNORECASE)
    # Curso embebido en el nombre del archivo (ej. simce2m2024_rbd.csv → '2m')
    regex_curso = re.compile(r'simce(2m|4b|6b|8b)\d{4}', re.IGNORECASE)

    columnas_deseadas = ['rbd', 'agno', 'curso', 'prom_mate_rbd', 'prom_lect_rbd',
                         'cod_grupo', 'cod_depe2', 'cod_rural_rbd']

    for archivo in archivos_rbd:
        print(f"Procesando: {os.path.basename(archivo)}")
        
        try:
            # 1. Detección robusta de delimitador usando sep=None y el motor de python
            try:
                df_temp = pd.read_csv(archivo, sep=None, engine='python', encoding='utf-8-sig')
            except UnicodeDecodeError:
                df_temp = pd.read_csv(archivo, sep=None, engine='python', encoding='latin1')
        except Exception as e:
            print(f"Error al leer {archivo}: {e}")
            continue
            
        # Normalizar los nombres de columnas a minúsculas
        df_temp.columns = [str(col).lower().strip() for col in df_temp.columns]
        
        # 2. Estandarización Dinámica de los promedios
        rename_dict = {}
        for col in df_temp.columns:
            if regex_mate.match(col):
                rename_dict[col] = 'prom_mate_rbd'
            elif regex_lect.match(col):
                rename_dict[col] = 'prom_lect_rbd'
                
        df_temp.rename(columns=rename_dict, inplace=True)

        # Inferir curso desde el nombre de archivo (el sufijo se pierde al renombrar mate/lect)
        if 'curso' not in df_temp.columns:
            match_curso = regex_curso.search(os.path.basename(archivo).lower())
            df_temp['curso'] = match_curso.group(1).lower() if match_curso else pd.NA

        # Insertar columna agno si no viene, derivándola del nombre del archivo si es necesario
        if 'agno' not in df_temp.columns:
            # Extraer el año del nombre del archivo (ej. simce2m2025_...)
            match_agno = re.search(r'20\d{2}', os.path.basename(archivo))
            if match_agno:
                df_temp['agno'] = int(match_agno.group())
            else:
                df_temp['agno'] = pd.NA

        # 3. Filtro Selectivo: Validar e incluir sólo aquellas columnas que existan en el df_temp
        cols_presentes = [col for col in columnas_deseadas if col in df_temp.columns]
        df_temp = df_temp[cols_presentes]
        
        # Llenar con NaN las columnas deseadas que no vinieron en este año específico
        for col_faltante in set(columnas_deseadas) - set(cols_presentes):
            df_temp[col_faltante] = pd.NA
            
        # Forzar el orden de las columnas
        df_temp = df_temp[columnas_deseadas]
        
        df_list.append(df_temp)

    if not df_list:
        print("No se pudo extraer información válida de los archivos.")
        return

    # Consolidación final
    df_consolidado = pd.concat(df_list, ignore_index=True)
    
    # Asegurar que los puntajes sean numéricos (SIMCE suele usar '*' o ' ' para datos faltantes)
    df_consolidado['prom_mate_rbd'] = pd.to_numeric(df_consolidado['prom_mate_rbd'], errors='coerce')
    df_consolidado['prom_lect_rbd'] = pd.to_numeric(df_consolidado['prom_lect_rbd'], errors='coerce')
    
    # 4. Limpieza de la Variable Objetivo (Dropna estricto)
    filas_iniciales = len(df_consolidado)
    df_consolidado.dropna(subset=['prom_mate_rbd', 'prom_lect_rbd'], how='any', inplace=True)
    filas_finales = len(df_consolidado)
    
    print(f"\n--- Resumen de Limpieza ---")
    print(f"Total registros obtenidos: {filas_iniciales}")
    print(f"Registros sin Target eliminados: {filas_iniciales - filas_finales}")
    print(f"Total registros limpios para modelo: {filas_finales}")
    
    # Asegurar tipos de datos lógicos
    df_consolidado['agno'] = pd.to_numeric(df_consolidado['agno'], errors='coerce').astype('Int64')
    
    # Guardar en CSV
    df_consolidado.to_csv(output_file, index=False, sep=',', encoding='utf-8')
    print(f"\nArchivo guardado exitosamente en: {output_file}")


if __name__ == "__main__":
    # Rutas dinámicas basadas en la ubicación de este script
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DIR_DATOS = os.path.join(BASE_DIR, "data", "raw", "simce")
    
    # Asegurar que exista la carpeta
    os.makedirs(os.path.join(BASE_DIR, "data", "processed"), exist_ok=True)
    OUTPUT_CSV = os.path.join(BASE_DIR, "data", "processed", "dataset_simce_consolidado.csv")
    
    consolidar_datos_simce(DIR_DATOS, OUTPUT_CSV)