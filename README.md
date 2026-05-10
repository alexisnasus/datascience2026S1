# 📚 Proyecto de Data Science: Predicción SIMCE mediante IDPS (2016-2025)

Este repositorio contiene el pipeline de datos (fase de *Obtain & Scrub* bajo la metodología OSEMN) para procesar el histórico de datos educacionales de Chile (SIMCE e IDPS). El objetivo final es preparar un **Dataset Maestro** consolidado para entrenar un modelo de Machine Learning orientado a predecir el puntaje SIMCE basándose en variables de contexto y desarrollo personal y social (IDPS).

## 📂 Estructura del Proyecto

El proyecto sigue el estándar de la industria para ciencia de datos:

- `data/`: Almacena los datos crudos (`raw/`), y los resultantes (`processed/`). *Nota: Los datos crudos no se suben a GitHub por su peso.*
- `src/`: Contiene el código fuente y los scripts de ETL y consolidación.
- `docs/`: Documentación, diccionarios de variables y metadatos.

## ⚙️ Configuración del Entorno (Local)

Para reproducir este proyecto en tu máquina local, sigue estos pasos para configurar tu entorno virtual en Windows e instalar sus dependencias:

1. **Crear el entorno virtual:**
   ```bash
   python -m venv venv
   ```
2. **Activar el entorno virtual:**
   ```bash
   venv\Scriptsctivate
   ```
3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Pipeline de Datos (ETL)

Durante la fase inicial, se lidiaron con anomalías temporales (*data drifts*), múltiples delimitadores y variaciones de formato ancho/largo. Se desarrollaron los siguientes scripts en `src/`:

### 1. Procesamiento de IDPS (`scrub_and_merge_idps.py`)
Limpia y estandariza múltiples archivos IDPS. Detecta delimitadores automáticamente, maneja el salto de formato *wide* a *long* en 2023 aplicando pivotes dinámicos, y fusiona archivos de manera horizontal (por año/curso) y vertical (histórico completo 2016-2025).

### 2. Procesamiento de SIMCE (`fusion_simce.py`)
Consolida, estandariza y limpia los archivos SIMCE estrictamente a nivel de establecimiento (`_rbd`). Maneja problemas de _encoding_ con caracteres latinos, normaliza los nombres de los promedios y aplica limpieza selectiva filtrando escuelas nulas en la variable *Target* (Lectura y Matemática).

### 3. Generación de Diccionarios y Glosas
- **`consolidar_glosas.py`**: Itera sobre diccionarios `.xlsx` del IDPS consolidando miles de registros en un archivo maestro de referencia.
- **`extraer_diccionario_completo.py`**: Escanea dinámicamente las hojas de cálculo de las glosas SIMCE evadiendo índices y metadatos basura para recuperar 168 variables únicas históricas.

## 📝 Documentación Adicional
En la capeta `docs/` se encuentran detallados los diccionarios oficiales (`diccionario_idps.md`, `simce.md`) que registran las predictoras base, metadatos, codificación categórica y registro de cambios estructurales documentados a lo largo de los años.
