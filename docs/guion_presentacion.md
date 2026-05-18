# Guión de presentación — Slides 9 a 13 (fase Obtain & Scrub)

> Tu bloque cubre el corazón técnico del proyecto: cómo se obtuvieron y limpiaron
> 10 años de datos. Es justo donde más trabajaste, así que dominar estos conceptos
> te da seguridad. Este archivo tiene, por cada slide: **qué se ve**, **los
> conceptos que debes entender**, un **guión hablado sugerido**, **el desfase**
> que debes mencionar de viva voz, y la **transición** a la siguiente.

---

## Vocabulario base (tenlo claro antes de empezar)

Si dominas estas 6 palabras, todo el bloque fluye:

- **SIMCE**: Sistema de Medición de la Calidad de la Educación. Prueba
  estandarizada chilena (Matemática y Lectura). Aquí la usamos a nivel de
  **establecimiento** (promedio del colegio), no por alumno. Es nuestra
  **variable objetivo** (lo que el modelo quiere predecir).
- **IDPS**: Indicadores de Desarrollo Personal y Social. Miden lo *no
  académico*: autoestima, convivencia, hábitos, participación. Son nuestras
  **variables predictoras**.
- **RBD**: Rol Base de Datos. El identificador único de cada colegio. Es la
  llave que permite cruzar SIMCE con IDPS.
- **Agencia de Calidad de la Educación**: organismo público chileno que
  produce y publica ambas bases. Todo el "drift" viene de que *ellos*
  cambiaron sus formatos año a año.
- **OSEMN**: la metodología del proyecto — **O**btain, **S**crub, **E**xplore,
  **M**odel, i**N**terpret. Tus slides son **Obtain** (conseguir y consolidar)
  y **Scrub** (limpiar). Aún no modelamos.
- **Data drift**: cuando la *estructura* de los datos cambia con el tiempo
  (nombres de columnas, formato, delimitador) aunque el fenómeno medido sea el
  mismo. Es el problema central que resuelve tu bloque.

---

## SLIDE 9 — Preparación de datos (cobertura y desafíos)

**En pantalla:** Cobertura (9 años 2016–2025 excluyendo pandemia; 4 niveles:
4°, 6°, 8° básico y II medio). Desafíos detectados: formatos mixtos (.csv y
.txt), tres delimitadores (coma, pipe `|`, punto y coma `;`), encoding
inconsistente (Latin-1 vs UTF-8). Y un "¿Por qué importa?".

**Conceptos que debes entender:**

- **Cobertura temporal**: 2016 a 2025 son 10 años calendario, pero **2020 y
  2021 no tienen datos porque no se rindió SIMCE por la pandemia** → por eso
  se habla de "excluyendo años de pandemia". Quedan 8 años con datos reales.
- **Delimitador**: el carácter que separa columnas en un archivo de texto.
  La Agencia usó tres distintos según el año: pipe `|` (2016–2022), punto y
  coma `;` (2023+) y coma. Si lo lees con el separador equivocado, **todas
  las columnas se fusionan en una sola** y el archivo queda inservible.
- **Encoding**: cómo se guardan los caracteres especiales (tildes, ñ).
  Latin-1 y UTF-8 codifican la "ñ" distinto; abrir un archivo con el encoding
  equivocado corrompe los nombres de columnas y comunas.
- **Por qué importa**: un solo archivo mal leído no es un error aislado —
  **contamina un año completo** (todos los colegios de ese año/curso) y rompe
  la trazabilidad histórica que es justo lo que buscamos.

**Guión hablado sugerido:**

> "Trabajamos con datos de la Agencia de Calidad de la Educación, cubriendo
> 2016 a 2025. Son nueve ventanas de tiempo, pero ojo: 2020 y 2021 no
> aparecen porque durante la pandemia no se rindió el SIMCE. Cubrimos cuatro
> niveles: 4°, 6° y 8° básico, y II medio.
>
> El desafío real no fue el volumen, sino la **heterogeneidad**: los archivos
> vienen en `.csv` y `.txt`, mezclan tres delimitadores distintos —coma, pipe
> y punto y coma— y dos codificaciones, Latin-1 y UTF-8. Esto importa porque
> no es un detalle cosmético: si un archivo se lee con el separador
> equivocado, *todas* sus columnas colapsan y se pierde un año entero de
> datos sin que el script avise. Por eso lo primero que construimos fue
> detección automática de formato."

**Desfase a mencionar (importante, hazlo aquí):**

> "Quiero hacer una aclaración: esta presentación refleja una versión anterior
> del pipeline. Desde entonces avanzamos bastante en Obtain y Scrub, así que
> en las próximas slides van a ver cifras que ya quedaron desactualizadas;
> las iré corrigiendo al pasar."

Decirlo aquí, al inicio de tu bloque, te cubre para todas las slides
siguientes y suena profesional, no como excusa.

**Transición:** "Y el desafío más grave no era el delimitador, sino un cambio
de estructura mucho más profundo…" → pasa a slide 10.

---

## SLIDE 10 — Drifts estructurales

**En pantalla:** Cambio de formato impuesto por la Agencia. Antes (2016–2022):
formato **Wide** — una columna por indicador IDPS (`ind_am`, `ind_cc`,
`ind_hv`, `ind_pf`). Desde 2023: formato **Long** — una columna `ind` con el
nombre del indicador y una columna `prom` con el valor ("filas derretidas").
Un "¿Por qué importa?".

**Conceptos que debes entender:**

- **Formato Wide ("ancho")**: cada indicador es su propia columna. Una fila
  por colegio, 4 columnas de indicadores. Es el formato "natural" para un
  modelo: cada columna = una variable.
- **Formato Long ("largo") / datos derretidos (melt)**: en vez de 4 columnas,
  hay 2: una que dice *cuál* indicador (`ind`) y otra con *el valor* (`prom`).
  Cada colegio ocupa 4 filas (una por indicador) en lugar de 1.
- **Pivot / unpivot**: la operación que transforma Long → Wide. Tomas la
  columna `ind`, la "abres" en 4 columnas y rellenas con `prom`. Esto es lo
  que el script hace para **homologar** ambos formatos a uno solo.
- **Por qué importa**: si no homologas, el modelo ve "el mismo indicador"
  como cosas distintas según el año, la matriz queda con **más del 80% de
  celdas vacías (NaN)** y es inutilizable para regresión.

**Guión hablado sugerido:**

> "El cambio más profundo lo impuso la propia Agencia en 2023. Hasta 2022 los
> datos venían en formato **wide**: cada indicador IDPS era una columna —
> autoestima, convivencia, hábitos, participación—, una fila por colegio.
>
> Desde 2023 cambiaron a formato **long**: en lugar de cuatro columnas, dejaron
> una sola columna que dice *qué* indicador es y otra con el valor. El mismo
> colegio que antes era una fila, ahora son cuatro filas 'derretidas'.
>
> Esto importa muchísimo: si juntáramos los años sin homologar la estructura,
> el modelo interpretaría el indicador de 2022 y el de 2023 como variables
> diferentes. La matriz quedaría con más del 80% de valores nulos —
> completamente inutilizable para una regresión. La solución fue pivotear todo
> al mismo formato wide antes de consolidar."

**Desfase a mencionar:**

> "Un detalle que descubrimos *después* de armar esta slide: 2025 introdujo
> *otro* cambio más — ya no usa el nombre del indicador en texto, sino un
> código numérico del 1 al 4. Tuvimos que mapear esos códigos a mano. Es un
> buen ejemplo de que el drift no para: cada año trae una sorpresa nueva."

**Transición:** "Con la estructura ya homologada, vino la limpieza propiamente
tal…" → slide 11.

---

## SLIDE 11 — Limpieza / Consolidación

**En pantalla:** A la izquierda, salida de terminal procesando archivos
`simce..._rbd...`. A la derecha: variable a predecir (Puntajes SIMCE de
Matemática y Lectura), 1. Limpieza del target (dropna estricto), 2. Estrategia
de fusión (merge horizontal, append vertical, deduplicación), 3. Output
(dataset maestro + diccionarios).

**Conceptos que debes entender:**

- **Variable objetivo / target (la "Y")**: lo que el modelo intenta predecir.
  Aquí son los puntajes SIMCE de Matemática y Lectura del colegio.
- **`dropna` estricto**: eliminar toda fila que no tenga puntaje SIMCE.
  La razón es conceptual: en **aprendizaje supervisado** el modelo aprende de
  ejemplos *con respuesta conocida*. Una fila sin Y no enseña nada; peor aún,
  **contamina la validación cruzada y sesga las métricas**.
- **Secreto estadístico**: la Agencia censura (deja en blanco) los puntajes de
  colegios con muy pocos alumnos, para que no se pueda identificar a un
  estudiante. Por eso muchas filas sin target *no son un error*: son datos
  protegidos por ley. Eso justifica eliminarlas sin culpa.
- **Merge horizontal (join)**: pegar columnas de dos tablas que comparten una
  llave —aquí `(rbd, año, curso)`—. Une indicadores con sub-dimensiones.
- **Append vertical (concat)**: apilar filas — consolidar los 8 años en una
  sola tabla histórica.
- **Deduplicación por llave compuesta**: si un colegio aparece dos veces para
  el mismo `(rbd, año, curso)`, se deja una sola fila. La llave es "compuesta"
  porque ninguna columna sola identifica una fila; se necesitan las tres.
- **Diccionario de datos**: documento que describe cada variable. Sirve para
  **gobernanza y reproducibilidad**: cualquiera puede entender y re-correr el
  pipeline sin adivinar qué significa cada columna.

**Guión hablado sugerido:**

> "Nuestra variable a predecir son los puntajes SIMCE de Matemática y Lectura.
> El primer paso de limpieza es un `dropna` estricto sobre el target: si una
> fila no tiene puntaje, se elimina. Esto no es opcional — en aprendizaje
> supervisado, una fila sin respuesta no aporta al entrenamiento y además
> ensucia la validación cruzada. Y hay una razón de fondo: muchos de esos
> blancos son colegios censurados por **secreto estadístico**, datos
> protegidos, no errores nuestros.
>
> La estrategia de fusión tiene tres movimientos: un *merge horizontal* que
> une indicadores y sub-dimensiones por colegio, año y curso; un *append
> vertical* que apila los ocho años de historia; y una deduplicación final
> por esa llave compuesta. El resultado son dos datasets consolidados —IDPS y
> SIMCE— más sus diccionarios de datos en Markdown, para que el pipeline sea
> reproducible y auditable."

**Desfase a mencionar (clave aquí — la terminal muestra cifras viejas):**

> "En la consola de la slide ven 'registros limpios: 53.657'. Esa cifra
> quedó obsoleta: justamente encontramos un bug en la detección de
> delimitador que estaba descartando en silencio los años 2016, 2017 y 2022.
> Al corregirlo, SIMCE pasó de 53 mil a cerca de **99 mil registros**, con los
> 8 años completos. O sea, recuperamos casi el doble de datos."

**Transición:** "Veamos entonces en qué quedó la fase completa…" → slide 12.

---

## SLIDE 12 — Resultados de la fase Obtain & Scrub

**En pantalla:** 1. Dataset IDPS consolidado (110.211 filas × 57 variables;
8 años; 4 niveles). 2. Dataset SIMCE consolidado (53.657 filas × 7 variables;
5 años). Siguiente paso: cruce IDPS × SIMCE por `(rbd, año)`. Un "¿Por qué
importa?".

**Conceptos que debes entender:**

- **Dataset consolidado**: el producto de Obtain — *todos* los años y niveles
  unificados en una sola tabla limpia, una fila por `(rbd, año, curso)`.
- **Separación Obtain / Scrub**: la idea de fondo del "¿por qué importa?".
  Tener las dos bases consolidadas y limpias **por separado** permite después
  recombinarlas con distintas estrategias (qué años incluir, cómo tratar
  nulos) **sin volver a tocar los archivos crudos**. Es *separación de
  responsabilidades*: Obtain no decide, solo consolida; Scrub experimenta.
- **Cruce / inner join por `(rbd, año)`**: el siguiente paso une IDPS con
  SIMCE solo donde *ambos* existen para el mismo colegio y año → dataset
  analítico final, el que alimenta el modelo.

**Guión hablado sugerido:**

> "El resultado de la fase son dos bases consolidadas e independientes. Por un
> lado IDPS, con todos los indicadores y variables de contexto; por otro
> SIMCE, con los puntajes. El paso siguiente, que ya está en curso, es
> cruzarlas por colegio y año para obtener el dataset analítico definitivo.
>
> ¿Por qué mantenerlas separadas y no fusionar todo de una? Porque tenerlas
> limpias por separado nos deja recombinarlas con distintas estrategias
> —qué años incluir, cómo tratar los nulos— sin volver nunca a los archivos
> crudos. Esa es la razón de separar Obtain de Scrub: Obtain solo consolida,
> las decisiones de limpieza fina se experimentan después, río abajo."

**Desfase a mencionar (esta slide es la que más cambió — sé claro):**

> "Las cifras de esta slide son de la versión preliminar y ya cambiaron las
> tres. IDPS: aplicamos una *whitelist* de columnas, así que en vez de 57
> variables —muchas eran ruido y duplicados— quedó en **20 columnas limpias**,
> con unas **108 mil filas**. SIMCE, con el fix del delimitador que mencioné,
> subió de 53 mil a **99 mil filas** y de 5 a **8 años** de cobertura. La
> estructura y la lógica son las mismas; lo que mejoró fue la cantidad y
> calidad de datos recuperados."

> Tip: ten estos tres números en la punta de la lengua — **20 columnas,
> 108 mil filas IDPS, 99 mil filas SIMCE, 8 años**. Confírmalos volviendo a
> correr los scripts antes de presentar, por si cambiaron levemente.

**Transición:** "Ahora, para esta primera iteración trabajamos con un
subconjunto acotado…" → slide 13.

---

## SLIDE 13 — Alcance del Dataset Preliminar

**En pantalla:** Estado actual (versión preliminar) para validar el pipeline
OSEMN y modelos base. Foco: 2° Medio, años 2017–2024. 11 columnas clave.
Definición de los 4 indicadores: `ind_am` (autoestima académica y motivación),
`ind_cc` (clima de convivencia), `ind_hv` (hábitos de vida saludable),
`ind_pf` (participación y formación ciudadana). Cierra: hay que escalar a
básica (4°, 6°, 8°) y a todo el histórico.

**Conceptos que debes entender:**

- **Dataset preliminar vs definitivo**: el preliminar es una *prueba de
  concepto* — un recorte pequeño (solo 2° Medio, 2017–2024) para validar que
  el pipeline OSEMN completo y los modelos base funcionan, antes de escalar.
- **Las 11 columnas clave (el "contrato")**: el notebook de modelado espera
  exactamente esas 11 columnas con esos nombres. Es un *contrato*: si Obtain
  cambia un nombre o el filtro de curso, el notebook se rompe en silencio.
- **Los 4 índices IDPS** (apréndetelos, te pueden preguntar):
  - `ind_am` — **Autoestima académica y motivación escolar**: cuánto cree el
    estudiante en su capacidad y sus ganas de aprender.
  - `ind_cc` — **Clima de convivencia escolar**: respeto, seguridad y
    ambiente de la sala.
  - `ind_hv` — **Hábitos de vida saludable**: alimentación, ejercicio,
    autocuidado.
  - `ind_pf` — **Participación y formación ciudadana**: involucramiento del
    estudiante en la vida escolar.
- **Escalabilidad**: el mismo cruce que se hizo para 2° Medio se debe extender
  a 4°, 6° y 8° básico y a todos los años → más registros, modelo más robusto.

**Guión hablado sugerido:**

> "Para esta primera iteración construimos un dataset *preliminar*: un recorte
> deliberadamente acotado —solo 2° Medio, entre 2017 y 2024— cuyo objetivo no
> es el modelo final, sino **validar que todo el pipeline OSEMN funciona de
> punta a punta**. De cientos de variables crudas destilamos 11 columnas
> clave: los puntajes, el contexto del colegio y los cuatro indicadores IDPS.
>
> Esos cuatro indicadores son: autoestima académica y motivación escolar;
> clima de convivencia; hábitos de vida saludable; y participación y formación
> ciudadana. Son nuestras variables predictoras: la hipótesis del proyecto es
> que el bienestar del estudiante explica parte del rendimiento académico.
>
> El paso natural —y aquí conecto con el avance real— es escalar este mismo
> cruce a básica y a todo el histórico."

**Desfase a mencionar (cierre fuerte de tu bloque):**

> "Y justamente eso último ya lo avanzamos. La slide habla de escalar 'a
> futuro', pero la consolidación de Obtain y Scrub que les mostré **ya cubre
> los cuatro niveles y los ocho años**. Lo que sigue pendiente es propagar esa
> cobertura ampliada al dataset analítico y al notebook de modelado, que
> todavía usa el recorte preliminar de 2° Medio. Así que el proyecto está más
> adelantado de lo que sugiere esta versión del PDF."

**Transición (entregas el bloque):** "Con la base de datos ya consolidada y
validada, pasamos al modelo propuesto…" → cede la palabra a quien sigue.

---

## El hilo narrativo (memoriza esta secuencia)

Tu bloque cuenta **una sola historia** en 5 pasos. Si te pierdes, vuelve a
este hilo:

1. **(S9)** Los datos venían sucios y heterogéneos → detección automática.
2. **(S10)** Además cambiaban de estructura cada año → homologación Wide.
3. **(S11)** Limpiamos el target y definimos cómo fusionar → consolidación.
4. **(S12)** Resultado: dos bases limpias e independientes.
5. **(S13)** Validamos con un recorte preliminar; ya lo estamos escalando.

Frase resumen si te quedas en blanco: *"Mi parte es cómo convertimos diez
años de archivos caóticos de la Agencia en una base limpia y reproducible
lista para modelar."*

---

## Preguntas probables (y cómo responder corto)

- **"¿Por qué no usan los años de pandemia?"** → Porque no se rindió SIMCE en
  2020–2021; no hay target que predecir, no es decisión nuestra.
- **"¿Por qué eliminan filas en vez de imputar el puntaje?"** → Es la variable
  objetivo: imputar la Y inventa la respuesta y sesga el modelo. Imputar se
  reserva para variables predictoras, no para el target.
- **"¿Qué pasó con las cifras del PDF?"** → Versión anterior; tras corregir un
  bug de delimitador recuperamos casi el doble de registros SIMCE y aplicamos
  una whitelist en IDPS. La lógica es la misma, mejoró la cobertura.
- **"¿IDPS realmente predice SIMCE?"** → Es la hipótesis a contrastar en la
  fase de modelado; nuestro bloque solo deja los datos listos para probarla.
- **"¿Por qué separar Obtain de Scrub?"** → Para poder experimentar distintas
  estrategias de limpieza sin re-procesar nunca los archivos crudos.
```
