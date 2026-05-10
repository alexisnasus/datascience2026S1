# Diccionario y Estructura de Datos - SIMCE (Nivel RBD)

Este documento describe la estructura consolidada de los datos del SIMCE a nivel de establecimiento (RBD). El objetivo principal de estos datos es servir de insumo para modelos de Machine Learning (Regresión) que predecirán el rendimiento académico.

## Variables Clave a Extraer

### Llaves Identificadoras
* **`rbd`**: Rol de Base de Datos, identificador único del establecimiento educacional.
* **`agno`**: Año de la evaluación SIMCE (2016-2025).

### Variable Objetivo (Target)
Los puntajes del SIMCE han sido estandarizados omitiendo el sufijo del curso para mantener la consistencia tabular:
* **`prom_mate_rbd`**: Puntaje promedio del establecimiento en la prueba de Matemática.
* **`prom_lect_rbd`**: Puntaje promedio del establecimiento en la prueba de Lectura (Comprensión de Lectura).

### Variables Predictoras de Contexto (Features)
* **`cod_grupo`**: Grupo Socioeconómico (GSE) asignado al establecimiento.
* **`cod_depe2`**: Código de dependencia administrativa del establecimiento (ej. Municipal, Particular Subvencionado, Particular Pagado).
* **`cod_rural_rbd`**: Índice de ruralidad del establecimiento (Urbano/Rural).

## Convención de Nombres de la Agencia de Calidad

En los archivos crudos originales provistos por la Agencia de Calidad, existe una convención dinámica donde el sufijo del curso evaluado se incrusta en el nombre de la variable de puntaje. 

Por ejemplo, para la prueba de matemática:
* 2° Medio $\rightarrow$ `prom_mate2m_rbd`
* 4° Básico $\rightarrow$ `prom_mate4b_rbd`
* 6° Básico $\rightarrow$ `prom_mate6b_rbd`
* 8° Básico $\rightarrow$ `prom_mate8b_rbd`

*Nota: Durante el proceso de "Scrub", nuestro pipeline normaliza automáticamente estas columnas eliminando la especificación del curso (`2m`, `4b`, `6b`, `8b`) hacia nombres genéricos (`prom_mate_rbd`, `prom_lect_rbd`). Esto es vital para evitar matrices dispersas (con valores NaN) derivadas de la segmentación por grado al consolidar el dataset.*

## Anexo: Universo Total de Variables Disponibles en Glosas
A continuación se documentan **todas** las variables originadas desde las glosas oficiales del SIMCE, incluyendo aquellas que fueron descartadas en la consolidación para el modelo predictivo.

| Variable (Genérica) | Descripción Oficial |
| :--- | :--- |
| `-1` | Diferencia negativa y estadísticamente significativa |
| `0` | Establecimientos rindieron Simce |
| `1` | En su totalidad ausentes y sin material aplicado |
| `2` | En su totalidad estudiantes integrados (NEEP) |
| `3` | En su totalidad ausentes y que se encuentran con algún cuestionario aplicado (alumnos y/o padres |
| `4` | En su totalidad estudiantes renuncian a responder |
| `5` | Establecimientos en paro |
| `6` | Con pérdida total de material en la aplicación |
| `7` | Comunidad educativa se niega a rendir |
| `agno` | Año de la evaluación |
| `cod_com` | Código de comuna |
| `cod_com_rbd` | Código de comuna del establecimiento |
| `cod_depe1` | Código de dependencia 6 categorías |
| `cod_depe2` | Código de dependencia 4 categorías |
| `cod_deprov` | Código de Departamento Provincial |
| `cod_deprov:rbd` | Nombre Deprov del establecimiento año 2015 |
| `cod_deprov_rbd` | Código  Deprov del establecimiento |
| `cod_grupo` | Código de grupo socioeconómico |
| `cod_pro` | Código de provincia |
| `cod_pro_rbd` | Código de provincia del establecimiento |
| `cod_reg` | Código de región |
| `cod_reg_rbd` | Código de región del establecimiento |
| `cod_rural_rbd` | Código de ruralidad del establecimiento |
| `codigo_bbdd` | Código único base de datos |
| `dependencia 1` | Sin descripción |
| `dependencia 2` | Sin descripción |
| `dif_hist_com` | Diferencia respecto a la evaluación anterior en Historia |
| `dif_hist_deprov` | Diferencia respecto a la evaluación anterior en Historia |
| `dif_hist_rbd` | Diferencia respecto a la evaluación anterior en Historia |
| `dif_hist_reg` | Diferencia respecto a la evaluación anterior en Historia |
| `dif_lect_com` | Diferencia respecto a la evaluación anterior en Lectura |
| `dif_lect_deprov` | Diferencia respecto a la evaluación anterior en Lectura |
| `dif_lect_rbd` | Diferencia respecto a la evaluación anterior en Lectura |
| `dif_lect_reg` | Diferencia respecto a la evaluación anterior en Lectura |
| `dif_mate_com` | Diferencia respecto a la evaluación anterior en Matemática |
| `dif_mate_deprov` | Diferencia respecto a la evaluación anterior en Matemática |
| `dif_mate_rbd` | Diferencia respecto a la evaluación anterior en Matemática |
| `dif_mate_reg` | Diferencia respecto a la evaluación anterior en Matemática |
| `dif_nat_com` | Diferencia respecto al año anterior en Ciencias Naturales |
| `dif_nat_deprov` | Diferencia respecto al año anterior en Ciencias Naturales |
| `dif_nat_rbd` | Diferencia respecto al año anterior en Ciencias Naturales |
| `dif_nat_reg` | Diferencia respecto al año anterior en Ciencias Naturales |
| `dif_soc_com` | Diferencia respecto al año anterior en Ciencias Sociales |
| `dif_soc_deprov` | Diferencia respecto al año anterior en Ciencias Sociales |
| `dif_soc_rbd` | Diferencia respecto al año anterior en ciencias sociales |
| `dif_soc_reg` | Diferencia respecto al año anterior en ciencias Sociales |
| `difgru_hist_rbd` | Diferencia con respecto al mismo GSE en Historia |
| `difgru_lect_rbd` | Diferencia con respecto al mismo GSE en Lectura |
| `difgru_mate_rbd` | Diferencia con respecto al mismo GSE en Matemática |
| `difgru_nat_rbd` | Diferencia con respecto al mismo GSE en Ciencias Naturales |
| `difgru_soc_rbd` | Diferencia con respecto al mismo GSE en Ciencias sociales |
| `dvrbd` | Dígito verificador RBD |
| `fecha_bbdd` | Fecha en la que se construyó esta bbdd |
| `grado` | Grado que rinde evaluación Simce |
| `gse` | Sin descripción |
| `marca: observaciones a las diferencias del puntaje` | Sin descripción |
| `marca: observaciones al puntaje` | Sin descripción |
| `marca_hist_rbd` | Razón por la que no se muestra el puntaje promedio Historia del establecimiento |
| `marca_lect_rbd` | Razón por la que no se muestra el puntaje promedio Lectura del establecimiento |
| `marca_mate_rbd` | Razón por la que no se muestra el puntaje promedio Matemática del establecimiento |
| `marca_nat_rbd` | Razón por la que no se muestra el puntaje promedio Ciencias Naturales del establecimient |
| `marca_soc_rbd` | Razón por la que no se muestra el puntaje promedio Ciencias sociales del establecimiento |
| `marcadif_hist_rbd` | Razón por la que no se muestran diferencias en Historia |
| `marcadif_lect_rbd` | Razón por la que no se muestran diferencias en Lectura |
| `marcadif_mate_rbd` | Razón por la que no se muestran diferencias en Matemática |
| `marcadif_nat_rbd` | Razón por la que no se muestra el puntaje en diferencias Ciencias Naturales |
| `marcadif_soc_rbd` | Razón por la que no se muestra el puntaje en diferencias Ciencias sociales |
| `nalu_hist_rbd` | Número de alumnos que rinde Historia |
| `nalu_lect_rbd` | Número de alumnos que rinde Lectura |
| `nalu_mate_rbd` | Número de alumnos que rinde Matemática |
| `nalu_nat_rbd` | Número de alumnos que rinde ciencias naturales |
| `nalu_soc_rbd` | Número de alumnos con puntaje Ciencias sociales |
| `no aplica` | Sin descripción |
| `noaplica` | Establecimiento no se considera en la aplicación |
| `nom_com` | Nombre de la comuna |
| `nom_com_rbd` | Nombre de la comunas del establecimiento |
| `nom_deprov` | Nombre de Departamento Provincial |
| `nom_deprov_rbd` | Nombre Deprov del establecimiento |
| `nom_pro` | Nombre de provincia |
| `nom_pro_rbd` | Nombre del departamento provincial del establecimiento |
| `nom_rbd` | Nombre del establecimiento |
| `nom_reg` | Nombre de región |
| `nom_reg_rbd` | Nombre de la región del establecimiento |
| `palu_eda_ade_lect_com` | Porcentaje de estudiantes con estándar adecuado en Lectura |
| `palu_eda_ade_lect_deprov` | Porcentaje de estudiantes con estándar adecuado en Lectura |
| `palu_eda_ade_lect_rbd` | Porcentaje de estudiantes con estándar adecuado en Lectura |
| `palu_eda_ade_lect_reg` | Porcentaje de estudiantes con estándar adecuado en Lectura |
| `palu_eda_ade_mate_com` | Porcentaje de estudiantes con estándar adecuado en Matemática |
| `palu_eda_ade_mate_deprov` | Porcentaje de estudiantes con estándar adecuado en Matemática |
| `palu_eda_ade_mate_rbd` | Porcentaje de estudiantes con estándar adecuado en Matemática |
| `palu_eda_ade_mate_reg` | Porcentaje de estudiantes con estándar adecuado en Matemática |
| `palu_eda_ele_lect_com` | Porcentaje de estudiantes con estándar elemental en Lectura |
| `palu_eda_ele_lect_deprov` | Porcentaje de estudiantes con estándar elemental en Lectura |
| `palu_eda_ele_lect_rbd` | Porcentaje de estudiantes con estándar elemental en Lectura |
| `palu_eda_ele_lect_reg` | Porcentaje de estudiantes con estándar elemental en Lectura |
| `palu_eda_ele_mate_com` | Porcentaje de estudiantes con estándar elemental en Matemática |
| `palu_eda_ele_mate_deprov` | Porcentaje de estudiantes con estándar elemental en Matemática |
| `palu_eda_ele_mate_rbd` | Porcentaje de estudiantes con estándar elemental en Matemática |
| `palu_eda_ele_mate_reg` | Porcentaje de estudiantes con estándar elemental en Matemática |
| `palu_eda_ins_lect_com` | Porcentaje de estudiantes con estándar insuficiente en Lectura |
| `palu_eda_ins_lect_deprov` | Porcentaje de estudiantes con estándar insuficiente en Lectura |
| `palu_eda_ins_lect_rbd` | Porcentaje de estudiantes con estándar insuficiente en Lectura |
| `palu_eda_ins_lect_reg` | Porcentaje de estudiantes con estándar insuficiente en Lectura |
| `palu_eda_ins_mate_com` | Porcentaje de estudiantes con estándar insuficiente en Matemática |
| `palu_eda_ins_mate_deprov` | Porcentaje de estudiantes con estándar insuficiente en Matemática |
| `palu_eda_ins_mate_rbd` | Porcentaje de estudiantes con estándar insuficiente en Matemática |
| `palu_eda_ins_mate_reg` | Porcentaje de estudiantes con estándar insuficiente en Matemática |
| `prom_hist_com` | Puntaje promedio comunal en Historia |
| `prom_hist_deprov` | Puntaje promedio DEPROV en Historia |
| `prom_hist_rbd` | Puntaje promedio del establecimiento en Historia |
| `prom_hist_reg` | Puntaje promedio regional en Historia |
| `prom_lect_com` | Puntaje promedio comunal en Lectura |
| `prom_lect_deprov` | Puntaje promedio DEPROV en Lectura |
| `prom_lect_rbd` | Puntaje promedio del establecimiento en Lectura |
| `prom_lect_reg` | Puntaje promedio regional en Lectura |
| `prom_mate_com` | Puntaje promedio comunal en Matemática |
| `prom_mate_deprov` | Puntaje promedio DEPROV en Matemática |
| `prom_mate_rbd` | Puntaje promedio del establecimiento en Matemática |
| `prom_mate_reg` | Puntaje promedio regional en Matemática |
| `prom_nat_com` | Puntaje promedio comunal en Ciencias Naturales |
| `prom_nat_deprov` | Puntaje promedio DEPROV en Ciencias Naturales |
| `prom_nat_rbd` | Puntaje promedio del establecimiento en ciencias naturales |
| `prom_nat_reg` | Puntaje promedio regional en Ciencias Naturales |
| `prom_sco_deprov` | Puntaje promedio DEPROV en Ciencias Sociales |
| `prom_soc_com` | Puntaje promedio comunal en Ciencias Sociales (o equivalente) |
| `prom_soc_deprov` | Puntaje promedio DEPROV en Ciencias Sociales |
| `prom_soc_rbd` | Puntaje promedio del establecimiento en Ciencias Sociales |
| `prom_soc_reg` | Puntaje promedio regional en Ciencias Sociales |
| `rbd` | Rol base de datos del establecimiento |
| `ruralidad` | Sin descripción |
| `sigdif_hist_com` | Indica si diferencia respecto a la evaluación anterior en Historia es significativa |
| `sigdif_hist_deprov` | Indica si diferencia respecto a la evaluación anterior en Historia es significativa |
| `sigdif_hist_rbd` | Indica si diferencia con la evaluación anterior en Historia es significativa |
| `sigdif_hist_reg` | Indica si diferencia respecto a la evaluación anterior en Historia es significativa |
| `sigdif_lect_com` | Indica si diferencia respecto a la evaluación anterior en Lectura es significativa |
| `sigdif_lect_deprov` | Indica si diferencia respecto a la evaluación anterior en Lectura es significativa |
| `sigdif_lect_rbd` | Indica si diferencia con la evaluación anterior en Lectura es significativa |
| `sigdif_lect_reg` | Indica si diferencia respecto a la evaluación anterior en Lectura es significativa |
| `sigdif_mate_com` | Indica si diferencia respecto a la evaluación anterior en Matemática es significativa |
| `sigdif_mate_deprov` | Indica si diferencia respecto a la evaluación anterior en Matemática es significativa |
| `sigdif_mate_rbd` | Indica si diferencia con la evaluación anterior en Matemática es significativa |
| `sigdif_mate_reg` | Indica si diferencia respecto a la evaluación anterior en Matemática es significativa |
| `sigdif_nat_com` | Indica si diferencia respecto al año anterior enCiencias Naturales es significativa |
| `sigdif_nat_deprov` | Indica si diferencia respecto al año anterior en Ciencias Naturales es significativa |
| `sigdif_nat_rbd` | Indica si diferencia con el año anterior en Ciencias naturales es significativa |
| `sigdif_nat_reg` | Indica si diferencia respecto al año anterior enCiencias Naturales es significativa |
| `sigdif_soc_com` | Indica si diferencia respecto al año anterior en Ciencias Sociales es significativa |
| `sigdif_soc_deprov` | Indica si diferencia respecto al año anterior en Ciencias Sociales es significativa |
| `sigdif_soc_rbd` | Indica si diferencia con el año anterior en Ciencias sociales es significativa |
| `sigdif_soc_reg` | Indica si diferencia respecto al año anterior en Ciencias Sociales es significativa |
| `siggru_hist_rbd` | Indica si diferencia con el mismo GSE en Historia es significativa |
| `siggru_lect_rbd` | Indica si diferencia con el mismo GSE en Lectura es significativa |
| `siggru_mate_rbd` | Indica si diferencia con el mismo GSE en Matemática es significativa |
| `siggru_nat_rbd` | Indica si diferencia con el mismo GSE en Ciencias naturales es significativa |
| `siggru_soc_rbd` | Indica si diferencia con el mismo GSE en Ciencias sociales es significativa |
| `significancia` | Sin descripción |
| `tabla 1. dependencia administrativa` | Sin descripción |
| `tabla 2. dependencia administrativa` | Sin descripción |
| `tabla 3. codigo de grupo socioeconómico` | Sin descripción |
| `tabla 4. código de ruralidad del establecimiento` | Sin descripción |
| `tabla 4. índice de ruralidad del establecimiento` | Sin descripción |
| `tabla 5.  marca: observaciones al puntaje` | Sin descripción |
| `tabla 5. marca: observaciones al puntaje` | Sin descripción |
| `tabla 6. marca: observaciones a las diferencias del puntaje` | Sin descripción |
| `tabla 7.  significancia` | Sin descripción |
| `tabla 7. no aplica` | Sin descripción |
| `tabla 8. no aplica` | Sin descripción |
| `valor` | Descripción |
