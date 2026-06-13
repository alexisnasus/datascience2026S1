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
python src/scrub_and_merge_idps.py
python src/fusion_simce.py

# 2. Armar el dataset analítico (elige el enfoque que quieras modelar)
python src/build_dataset_historico.py            # contemporáneo, todos los cursos juntos
python src/build_dataset_historico_per_curso.py  # contemporáneo, un archivo por curso
python src/build_dataset_maestro.py              # predictivo (desfase temporal T-1)

# 3. Modelar
python src/datascienceproyecto1.py               # regresión, todos los cursos juntos
python src/datascienceproyecto1_per_curso.py     # regresión, un modelo por curso
```

## 📂 Estructura

- **`src/`** — todos los scripts (ver tabla más abajo).
- **`data/agrupado/`** — fuente cruda canónica (versionada: el pipeline corre en un clon limpio sin pasos manuales).
- **`data/processed/`** — salidas del pipeline (consolidados + datasets analíticos). `por_curso/` guarda los 4 datasets por nivel.
- **`data/glosas_idps/`, `data/glosa_simce/`** — diccionarios oficiales `.xlsx` de variables.
- **`docs/`** — diccionarios de variables y notas (`diccionario_*.md`).
- **`reports/`** — figuras generadas por los modelos.

## 🧩 Scripts (`src/`)

| Script | Qué hace |
|---|---|
| **Consolidación (Obtain & Scrub)** | |
| `scrub_and_merge_idps.py` | Limpia y consolida los IDPS de todos los años/cursos → `dataset_consolidado_idps.csv`. Maneja el cambio de formato ancho→largo (2023+) y los códigos de indicador. |
| `fusion_simce.py` | Consolida los SIMCE a nivel establecimiento → `dataset_simce_consolidado.csv`. Detecta delimitadores y _encoding_ por año y normaliza los nombres de los puntajes. |
| **Join (datasets analíticos)** | |
| `build_dataset_historico.py` | Une IDPS+SIMCE del **mismo año**, todos los cursos → `dataset_historico_completo.csv`. |
| `build_dataset_historico_per_curso.py` | Mismo join contemporáneo, pero **un archivo por curso** (4b/6b/8b/2m) en `data/processed/por_curso/`. |
| `build_dataset_maestro.py` | Une con **desfase T-1** (predice el SIMCE del año siguiente) → `dataset_maestro.csv`. |
| **Modelado** | |
| `datascienceproyecto1.py` | Regresión lineal (Matemática y Lectura) con **todos los cursos juntos**. Figuras en `reports/figuras_regresion_lineal/`. |
| `datascienceproyecto1_per_curso.py` | Regresión lineal **por curso** (8 modelos) para ver cómo cambia el efecto de los IDPS según el nivel. Figuras en `reports/figuras_por_curso/`. |
| **Utilidad** | |
| `extract_glossaries.py` | Extrae las glosas/diccionarios oficiales `.xlsx` → `todas_las_glosas*.csv`. |

> **Legacy / en pausa (no usar):** `ridge_simce_modelo_unificado.py` (modelo Ridge, aún en formato Colab, en pausa), `extraccion_presentacion.py` (huérfano: leía un PDF ya eliminado) y los artefactos congelados `data/dataset_historico_final.csv` + `reports/figuras/`.

## 📝 Documentación

La carpeta `docs/` detalla los diccionarios de variables (`diccionario_idps.md`, `diccionario_simce.md`),
los datasets analíticos (`diccionario_historico_completo.md`, `diccionario_maestro.md`) y el registro de
los cambios estructurales año a año que motivaron gran parte de la limpieza.
