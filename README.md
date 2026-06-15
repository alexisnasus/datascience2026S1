# 📚 SIMCE × IDPS — Pipeline de datos y modelos (2016–2025)

Proyecto de Data Science (metodología **OSEMN**) que consolida 10 años de datos educacionales
chilenos — puntajes **SIMCE** e **Indicadores de Desarrollo Personal y Social (IDPS)** — y entrena
modelos de regresión para estudiar **cómo se relaciona el desarrollo personal/social con el
rendimiento académico** de los establecimientos. Todo el código, los datos y la documentación están
en **español**.

## ⚙️ Levantar el entorno

Crea y activa un entorno virtual, luego instala las dependencias:

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (cmd)**
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 🚀 Cómo correr el pipeline

El orden es **consolidar → unir → modelar**. Cada script resuelve sus propias rutas, así que
funcionan desde cualquier carpeta. No hay tests ni build: "correr" es ejecutar estos scripts.

```bash
# 1. Consolidar las dos fuentes crudas
python src/obtener_y_limpiar/scrub_and_merge_idps.py
python src/obtener_y_limpiar/fusion_simce.py

# 2. Armar el dataset analítico (elige el enfoque que quieras modelar)
python src/construir_datasets/build_dataset_historico.py            # contemporáneo, todos los cursos juntos
python src/construir_datasets/build_dataset_historico_per_curso.py  # contemporáneo, un archivo por curso
python src/construir_datasets/build_dataset_maestro.py              # predictivo (desfase temporal T-1)

# 3. Modelar (dos documentos maestros, uno por enfoque temporal)
python src/modelado/modelo_contemporaneo_simce.py                  # contemporáneo: global + por curso + unificado
python src/modelado/modelo_predictivo_simce.py                     # predictivo: desfase T-1 (out-of-time)
```

## 📂 Estructura

- **`src/`** — scripts vigentes, organizados por fase: `obtener_y_limpiar/`, `construir_datasets/`, `modelado/` y `utilidades/` (ver tabla más abajo).
- **`data/agrupado/`** — fuente cruda canónica (versionada: el pipeline corre en un clon limpio sin pasos manuales).
- **`data/processed/`** — salidas del pipeline (consolidados + datasets analíticos). `por_curso/` guarda los 4 datasets por nivel.
- **`data/glosas_idps/`, `data/glosa_simce/`** — diccionarios oficiales `.xlsx` de variables.
- **`docs/`** — diccionarios de variables y notas (`diccionario_*.md`).
- **`reports/`** — figuras generadas por los maestros (`figuras_contemporaneo/{global,por_curso,unificado}/` y `figuras_predictivo/`).
- **`legacy/`** — versiones anteriores archivadas (no eliminadas): espeja el repo con `legacy/src/` (scripts del 1er entregable) y `legacy/reports/` (sus figuras).

## 🧩 Scripts (`src/`)

| Script | Qué hace |
|---|---|
| **Consolidación (Obtain & Scrub) · `src/obtener_y_limpiar/`** | |
| `scrub_and_merge_idps.py` | Limpia y consolida los IDPS de todos los años/cursos → `dataset_consolidado_idps.csv`. Maneja el cambio de formato ancho→largo (2023+) y los códigos de indicador. |
| `fusion_simce.py` | Consolida los SIMCE a nivel establecimiento → `dataset_simce_consolidado.csv`. Detecta delimitadores y _encoding_ por año y normaliza los nombres de los puntajes. |
| **Construir datasets (Join) · `src/construir_datasets/`** | |
| `build_dataset_historico.py` | Une IDPS+SIMCE del **mismo año**, todos los cursos → `dataset_historico_completo.csv`. |
| `build_dataset_historico_per_curso.py` | Mismo join contemporáneo, pero **un archivo por curso** (4b/6b/8b/2m) en `data/processed/por_curso/`. |
| `build_dataset_maestro.py` | Une con **desfase T-1** (predice el SIMCE del año siguiente) → `dataset_maestro.csv`. |
| **Modelado · `src/modelado/`** (maestros `.py` con celdas `# %%`, corren local) | |
| `modelo_contemporaneo_simce.py` | Análisis **contemporáneo** (mismo año) en 3 secciones: **global** (todos los cursos), **por curso** (8 modelos, IQR por curso) y **unificado** (mate+lect con `tipo_prueba`). Figuras en `reports/figuras_contemporaneo/`. |
| `modelo_predictivo_simce.py` | Análisis **predictivo** (desfase T-1, *out-of-time*): OLS/Ridge/ElasticNet, dos modelos. Figuras en `reports/figuras_predictivo/`. |
| **Utilidad · `src/utilidades/`** | |
| `extract_glossaries.py` | Extrae las glosas/diccionarios oficiales `.xlsx` → `todas_las_glosas*.csv`. |

> **Archivado en `legacy/` (no usar, no eliminado):** los scripts del 1er entregable y sus figuras viven en `legacy/src/` (`datascienceproyecto1.py`, `datascienceproyecto1_per_curso.py`, `datascienceproyecto_modelounificado_beta.py`, `ridge_simce_modelo_unificado.py`, y el huérfano `extraccion_presentacion.py` que leía un PDF ya eliminado) y `legacy/reports/`; los dos maestros absorben sus análisis. El artefacto congelado `data/dataset_historico_final.csv` también queda fuera de uso.

## 📝 Documentación

La carpeta `docs/` detalla los diccionarios de variables (`diccionario_idps.md`, `diccionario_simce.md`),
los datasets analíticos (`diccionario_historico_completo.md`, `diccionario_maestro.md`) y el registro de
los cambios estructurales año a año que motivaron gran parte de la limpieza.
