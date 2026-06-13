# Diccionario: `dataset_historico_completo.csv`

Dataset analítico **contemporáneo** (mismo año) generado por
[`src/build_dataset_historico.py`](../src/build_dataset_historico.py). Es la entrada de los
**modelos descriptivo-explicativos** (regresión lineal
[`src/datascienceproyecto1.py`](../src/datascienceproyecto1.py); Ridge
[`src/ridge_simce_modelo_unificado.py`](../src/ridge_simce_modelo_unificado.py)).

> **Contemporáneo vs. predictivo:** aquí predictores y target son del **mismo año T** (análisis de
> asociación, no de pronóstico). Para el enfoque predictivo con desfase temporal T-1 ver
> [`diccionario_maestro.md`](diccionario_maestro.md) (`dataset_maestro.csv`).
>
> **Variante por curso:** [`src/build_dataset_historico_per_curso.py`](../src/build_dataset_historico_per_curso.py)
> genera el **mismo dataset separado por nivel** en `data/processed/por_curso/dataset_historico_{4b,6b,8b,2m}.csv`
> (idéntico esquema; sus filas son las rebanadas por curso de este dataset). Es la entrada del modelo
> per-curso [`src/datascienceproyecto1_per_curso.py`](../src/datascienceproyecto1_per_curso.py), que
> recorta outliers (IQR) por curso.

## Estructura

Una fila por `(rbd, agno, curso)`. Reemplaza al antiguo `dataset_historico_final.csv` (que era
**solo 2° medio**); este incluye **todos los cursos** (2m, 4b, 6b, 8b).

| Columna | Tipo | Significado |
|---|---|---|
| `rbd` | int | Rol Base de Datos del establecimiento (clave) |
| `agno` | int | Año de medición (clave) |
| `curso` | str | Nivel evaluado: `2m`, `4b`, `6b`, `8b` (clave) |
| `prom_mate2m_rbd` | float | Puntaje SIMCE Matemática del establecimiento (nombre heredado; aplica a todos los cursos) |
| `prom_lect2m_rbd` | float | Puntaje SIMCE Lectura del establecimiento (nombre heredado) |
| `cod_grupo` | str | NSE: `Bajo` / `Medio bajo` / `Medio` / `Medio alto` / `Alto` |
| `cod_rural_rbd` | str | Ruralidad: `Urbano` / `Rural` |
| `cod_depe2` | str | Dependencia: `Municipal` / `Particular subvencionado` / `Particular pagado` / `SLEP` |
| `ind_am` | float | IDPS Autoestima Académica y Motivación (0-100) |
| `ind_cc` | float | IDPS Clima de Convivencia Escolar (0-100) |
| `ind_hv` | float | IDPS Hábitos de Vida Saludable (0-100) |
| `ind_pf` | float | IDPS Participación y Formación Ciudadana (0-100) |

## Decisiones de construcción

- **Join contemporáneo:** *inner join* de IDPS y SIMCE sobre `(rbd, agno, curso)` del **mismo año**.
- **Nombres heredados:** los puntajes mantienen el sufijo `2m` (`prom_mate2m_rbd`/`prom_lect2m_rbd`)
  y los indicadores el prefijo `ind_` para que los scripts de modelado existentes no requieran
  cambios al leer columnas. El sufijo `2m` es un nombre histórico: el dataset cubre los 4 cursos.
- **Contexto normalizado a texto canónico:** las fuentes mezclan códigos (`'1'`, `'1.0'`) y texto
  (`'Bajo'`, `'Urbano'`) según el año; aquí se entregan ya como etiquetas legibles.
- **Filtro de validez de puntaje:** se descartan las filas con `prom_mate2m_rbd` o `prom_lect2m_rbd`
  bajo `UMBRAL_SIMCE_MIN = 100` (la escala SIMCE va de ~100 a ~400). Eran ~617 filas con puntaje 0,
  errores de digitalización de la fuente. Es un filtro de **validez de dato**, distinto del recorte
  estadístico IQR que aplica el modelo.
- **No se imputa, escala ni eliminan nulos/outliers IQR:** el `dropna()` y el IQR viven en los
  scripts de modelado.

## Cobertura resultante

- **Filas:** ~97.928 (tras descartar ~617 puntajes inválidos) · **Cursos:** 4b (~39.706),
  2m (~20.646), 6b (~19.833), 8b (~17.743).
- **Años:** 2016-2025 (sin 2020-2021 por COVID, ausentes en la fuente).
- **Nulos en IDPS:** ~2,5% de las filas tiene algún `ind_*` faltante (los maneja el `dropna()` del modelo).
