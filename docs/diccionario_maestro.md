# Diccionario: `dataset_maestro.csv`

Dataset analítico **predictivo temporal** generado por
[`src/build_dataset_maestro.py`](../src/build_dataset_maestro.py). Es la entrada de la fase
**Explore & Model** ([`src/modelo_predictivo_simce.py`](../src/modelo_predictivo_simce.py) —
script `.py` con celdas `# %%`, convertible a notebook).

## Estructura

Una fila por `(rbd, agno=T, curso)`. Los **predictores se miden en el año T-1** y el **target en el
año T** (desfase de un año → enfoque predictivo, no contemporáneo).

| Columna | Tipo | Origen / significado |
|---|---|---|
| `rbd` | int | Rol Base de Datos del establecimiento (clave) |
| `agno` | int | **Año-objetivo T** (año del SIMCE que se predice) |
| `curso` | str | Nivel evaluado (`2m`, `4b`) |
| `curso_ord` | int | Curso como **ordinal** por años de escolaridad: `4b=4, 6b=6, 8b=8, 2m=10` |
| `nse_ord` | int | **NSE ordinal** 1..5 (`1=Bajo, 2=Medio bajo, 3=Medio, 4=Medio alto, 5=Alto`) |
| `cod_grupo` | str | Etiqueta NSE legible (derivada de `nse_ord`; referencia) |
| `cod_rural_rbd` | str | Ruralidad: `Urbano` / `Rural` (nominal) |
| `cod_depe2` | str | Dependencia: `Municipal` / `Particular subvencionado` / `Particular pagado` / `SLEP` (nominal) |
| `idps_am_prev` | float | IDPS **Autoestima Académica y Motivación** en T-1 (0-100) |
| `idps_cc_prev` | float | IDPS **Clima de Convivencia Escolar** en T-1 (0-100) |
| `idps_hv_prev` | float | IDPS **Hábitos de Vida Saludable** en T-1 (0-100) |
| `idps_pf_prev` | float | IDPS **Participación y Formación Ciudadana** en T-1 (0-100) |
| `simce_mate_prev` | float | SIMCE Matemática en T-1 (predictor autorregresivo) |
| `simce_lect_prev` | float | SIMCE Lectura en T-1 (predictor autorregresivo) |
| `target_mate` | float | **Target** SIMCE Matemática en T |
| `target_lect` | float | **Target** SIMCE Lectura en T |

## Decisiones de construcción

- **Desfase (lag):** se desplaza el año de las fuentes (`agno + 1`) y se hace *inner join* de
  `IDPS(T-1)`, `SIMCE(T-1)` y `SIMCE(T)` sobre `(rbd, agno, curso)`. El *inner join* garantiza
  años consecutivos y excluye automáticamente el salto **2019→2022** (sin datos 2020-21 por COVID).
- **NSE ordinal:** las fuentes mezclan códigos numéricos (`1..5`) y etiquetas de texto según el
  año; ambos se normalizan a `nse_ord` 1..5 (One-Hot perdería el orden, por eso ordinal).
- **No se imputa, escala ni eliminan outliers** en este script: esas decisiones (Scrub de
  modelado) viven en el `Pipeline` del notebook para **evitar fuga de información**.

## Cobertura resultante

- **Filas:** ~39.915 · **Cursos:** `4b` (~25.272) y `2m` (~14.643).
- **Años-objetivo (T):** 2017, 2018, 2023, 2024, 2025 (2018 solo aporta `2m`).
- **Partición temporal sugerida:** Train = `T ∈ {2017, 2018, 2023, 2024}`, Test (Out-of-Time) = `T = 2025`.

> **Importante — por qué solo `4b` y `2m`:** el SIMCE de **6° y 8° básico se rinde en años
> alternados** (`6b`: 2016/2018/2024; `8b`: 2017/2019/2025), por lo que **no existen pares de años
> consecutivos** para ellos y el desfase estricto T-1 los excluye. Solo `4b` y `2m` se evalúan de
> forma (casi) anual y permiten el enfoque predictivo. Para incorporar 6b/8b habría que renunciar al
> desfase (modelo contemporáneo) o usar un lag de 2 años con seguimiento de cohorte (no implementado).
