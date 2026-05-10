# Diccionario de Datos: IDPS y SIMCE (2016-2025)

## 📌 Contexto del Proyecto
Conjunto de datos histórico (2016-2025) que reúne los **Indicadores de Desarrollo Personal y Social (IDPS)** y variables socioeconómicas. El objetivo principal de este dataset maestro es entrenar un modelo de Machine Learning de regresión para predecir el rendimiento académico estandarizado (puntajes SIMCE).

---

## 🗃️ Diccionario de Variables Clave (Target y Predictoras Base)
Las siguientes variables se han estandarizado transversalmente sin importar el año o curso de origen.

| Variable Estandarizada | Tipo de Dato | Descripción |
|-------------------|-------------|-------------|
| `rbd` | Entero | Rol Base de Datos. Identificador único del establecimiento educacional. |
| `agno` | Entero | Año en que se rindió la evaluación (Ej. 2016 a 2025). |
| `curso` | Categórico | Nivel escolar evaluado (Ej. `2m`, `4b`, `6b`, `8b`). |
| `cod_depe2` | Categórico | Código de Dependencia del establecimiento (Municipal, Particular Subvencionado, Particular Pagado, etc.). |
| `cod_grupo` | Categórico | Grupo socioeconómico del establecimiento (GSE: Bajo, Medio Bajo, Medio, Medio Alto, Alto). |
| `cod_rural_rbd` | Binario/Cat | Indicador de ruralidad del establecimiento (0: Urbano, 1: Rural). |
| `simce_mate` | Flotante | **TARGET:** Puntaje SIMCE promedio del colegio en Matemáticas. |
| `simce_lect` | Flotante | **TARGET:** Puntaje SIMCE promedio del colegio en Lectura (Lenguaje). |
| `idps_am` | Flotante | Índice de Autoestima Académica y Motivación Escolar. |
| `dim_am_aa` | Flotante | Sub-dimensión: Autoestima Académica. |
| `dim_am_me` | Flotante | Sub-dimensión: Motivación Escolar. |
| `idps_cc` | Flotante | Índice de Clima de Convivencia Escolar. |
| `dim_cc_ao` | Flotante | Sub-dimensión: Ambiente Organizado. |
| `dim_cc_ar` | Flotante | Sub-dimensión: Ambiente de Respeto. |
| `dim_cc_as` | Flotante | Sub-dimensión: Ambiente Seguro. |
| `idps_hv` | Flotante | Índice de Hábitos de Vida Saludable. |
| `dim_hv_ac` | Flotante | Sub-dimensión: Actividad Física. |
| `dim_hv_ha` | Flotante | Sub-dimensión: Hábitos Alimenticios. |
| `dim_hv_va` | Flotante | Sub-dimensión: Vida Activa / Autocuidado. |
| `idps_pf` | Flotante | Índice de Participación y Formación Ciudadana. |
| `dim_pf_pa` | Flotante | Sub-dimensión: Participación. |
| `dim_pf_sp` | Flotante | Sub-dimensión: Sentido de Pertenencia. |
| `dim_pf_vd` | Flotante | Sub-dimensión: Vida Democrática. |
| `nom_rbd` | String | Nombre del establecimiento. (Variable descriptiva) |
| `cod_reg_rbd`| Entero | Código de la Región del establecimiento. (Variable geográfica) |
| `cod_pro_rbd`| Entero | Código de la Provincia del establecimiento. (Variable geográfica) |
| `cod_com_rbd`| Entero | Código de la Comuna del establecimiento. (Variable geográfica) |
| `codigo_bbdd`| String | Código interno BBDD / versión (Usado típicamente en 2023+). |
| `fecha_bbdd` | Fecha/Num | Sello temporal de la creación de la base de datos. |

### 📖 Codificación de Variables Categóricas

**Dependencia (`cod_depe2`):**
- `1`: Municipal
- `2`: Particular subvencionado
- `3`: Particular pagado
- `4`: SLEP (Servicios Locales de Educación Pública - incorporado en años recientes)

**Grupo Socioeconómico - GSE (`cod_grupo`):**
- `1`: Bajo
- `2`: Medio bajo
- `3`: Medio
- `4`: Medio alto
- `5`: Alto

**Ruralidad (`cod_rural_rbd`):**
- `1`: Urbano
- `2`: Rural

**Niveles Educativos (`curso` / `grado`):**
- `4b`: 4° básico
- `6b`: 6° básico
- `8b`: 8° básico
- `2m`: II medio

**Nombres de Indicadores IDPS (cuando vienen en formato Long bajo la columna `ind`):**
- `AM` o `AA`: Autoestima Académica y Motivación Escolar
- `CC`: Clima de Convivencia Escolar
- `HV`: Hábitos de Vida Saludable
- `PF`: Participación y Formación Ciudadana

---

## ⚠️ Registro de Variaciones Históricas conocidas (Data Drifts Estructurales)

1. **Estructura "Wide" vs "Long" (Derretida):**
   * **2016-2022:** Los datos presentaban una estructura Wide. Los índices de cada dimensión IDPS eran columnas distintas (ej. `ind_am`, `ind_cc`).
   * **2023-2025 (`idps2m2023_rbd_final.csv`):** Los datos presentan una estructura Long. Existe una columna llamada `ind` que contiene el nombre del indicador (AM, CC, HV, PF) y otra columna `prom` con el valor, por lo que existen filas duplicadas por colegio.

2. **Diferencias en Nombres de Columnas:**
   * **RBD vs rbd:** En 2018 los nombres provienen en Mayúsculas (`RBD`, `NOM_RBD`, `ind_am_rbd`).
   * **GSE / Dependencia:** Columnas presentes con nombres `cod_depe2`, `cod_grupo` y `cod_rural_rbd` en ciertos datasets; en versiones muy tempranas no se incluían todas.

3. **Formatos y Delimitadores (`.txt` y `.csv`):**
   * 2016: Archivos en formato `.txt` o `.csv` separados por **comas (`,`)**.
   * 2018: Archivos separados por **pipes (`|`)**.
   * 2023 en adelante: Archivos separados por **punto y comas (`;`)**.