# Explorar y enmascarar corpus Parquet

Un único script de experimentación llama a la CLI existente, sin duplicar la lógica:

```bash
python scripts/inspect_corpus.py train_full.parquet \
  --language cs --output-dir outputs/czech-preview \
  --techniques random terms whole_entities domain keyphrase \
  --terms 'diabetem' 'chronická pankreatitida' \
  --probability 0.2 --show-examples
```

Para varios corpus, se conservan sus filas y archivos por separado:

```bash
python scripts/inspect_corpus.py spanish.parquet czech.parquet \
  --languages es cs --output-dir outputs/multilingual-preview \
  --techniques all --terms-file initial_terms.txt \
  --probability 0.15 --show-examples --examples 5
```

Primero imprime documentos, frases, palabras elegibles, caracteres y entidades, por corpus
 y en total. Después ejecuta las técnicas indicadas. Sin `--techniques`, en una terminal
interactiva permite elegir después de consultar las estadísticas; en ejecución no
interactiva termina mostrando las estadísticas. Los ejemplos requieren `--show-examples`:
el texto parcialmente enmascarado todavía puede contener información clínica sensible.

## Entradas y segmentación

Son obligatorias `doc_id` y `text`. `entities_json` y `n_entities` son opcionales. Si solo
hay entidades, se calcula su contador; un contador positivo sin offsets es un error.
Se validan los offsets, el contenido y la correspondencia del contador con las entidades.
Los metadatos y las anotaciones existentes se conservan: no se usa la preparación de
corpus sin gold del entrenamiento. Los IDs pueden repetirse entre corpus diferentes;
deben ser únicos dentro de cada uno. Los corpus vacíos se rechazan explícitamente.

Se reutilizan `core.split_into_sentences` y `core.split_into_words`: pysbd, los modelos
Punkt incluidos e ICU cubren todos los códigos ISO 639-1 que admite ner_lab. El checo es
`cs`, no `cz`. La columna `language` tiene prioridad; `--languages` asigna un fallback a
cada archivo y `--language` un fallback común. Nunca se adivina el idioma por el nombre.

Las palabras elegibles son segmentos UAX #29 con al menos una letra o número Unicode;
la puntuación y el espacio no se enmascaran. Los offsets son caracteres `[start,end)` del
texto original y se guardan tanto para frases como para palabras. No se concatenan documentos.

## Técnicas y presupuesto

`random`/M0 selecciona palabras; `terms` prioriza coincidencias literales de términos
iniciales, incluyendo expresiones de varias palabras. Se compara por palabras completas
con Unicode casefold; `--case-sensitive` activa distinción de mayúsculas. No se interpretan
los términos como expresiones regulares. En `--terms-file` se admite un término por línea.

`entities`/M1 prioriza palabras de las anotaciones; `whole_entities`/M2 conserva entidades
como unidades indivisibles, fusionando los solapamientos. Es un análisis explícito de las
anotaciones, no una predicción de un modelo NER. Si no hay anotaciones se avisa: la señal
es cero y solo otras señales y el backfill habilitado pueden seleccionar palabras.

`domain`/M3 usa TF-IDF local; `keyphrase`/M4 utiliza candidatos léxicos locales;
`ner_domain`/M5, `ner_keyphrase`/M6 y `triple`/M7 fusionan sus señales con pesos iguales
experimentales. `entropy`/M8 calcula incertidumbre real con un checkpoint local indicado
por `--base-model`. `all` omite M8 si no se proporciona modelo y `terms` si no hay términos,
informándolo. Ninguna técnica no disponible se simula.

La selección opera sobre palabras, no subwords, y no es una corrupción de entrenamiento
80/10/10: cada palabra seleccionada se representa con `[MASK]` para inspección visual.
El presupuesto global es `floor(probability * total_palabras_elegibles)`. `0.2` significa
20%, `0.15` significa 15%. Por defecto las señales priorizan y se completa el presupuesto
con backfill. `--no-backfill` restringe la selección a evidencia positiva; si no hay
suficientes candidatos, el modo estricto falla. `--budget-mode underfill` permite una
selección menor, informando la desviación. Una frase inicial puede seleccionarse por palabras
independientes; la indivisibilidad corresponde a `whole_entities`.

**Distribución entre corpus:** la cuota ideal es B × palabras_corpus / palabras_totales.
Se usa resto mayor cuando esas cuotas enteras son factibles. Con restricciones de entidades,
se busca una asignación factible que minimice la suma de desviaciones cuadráticas respecto
a las cuotas ideales, conservando B si es posible. En underfill se maximiza primero el
presupuesto factible. Se guardan cuota ideal y asignación realizada por corpus. El programa
dinámico de casos restringidos puede ser costoso con presupuestos muy grandes; el caso
habitual de palabras usa la asignación directa.

Los scores se calculan para explorar los corpus suministrados: si incluyen dev/test,
no deben reutilizarse como estadísticas ajustadas para entrenamiento.

## Resultados y API

El directorio del script contiene `statistics.yaml` y `masking.yaml`, ejecutables también
con `lab run`, y los directorios de resultados correspondientes:

- `statistics/statistics.json`: agregados globales, por corpus, idiomas y segmentadores.
- `statistics/segmentation.parquet`: offsets documentales de frases y palabras.
- `masking/<tecnica>/corpus_000.parquet`, `corpus_001.parquet`, etc.: las filas originales,
  más `masked_text`, `n_masked_words`, `masked_word_spans_json`.
- `masking/<tecnica>/summary.json`: palabras seleccionadas, proporción realizada, cuota
  por corpus, backfill y desviación respecto a B.
- `masking/masking_config.json`: técnicas, términos y parámetros reproducibles.

No se sobrescribe un directorio existente. Los ejemplos se limitan en número y caracteres.
Se puede componer desde Python mediante `lab.cpt.inspect_corpora(...)`, o llamar por
separado a lectura, segmentación, señales, selección y persistencia, bajo
`lab.cpt.inspection`. No se modifica el comportamiento de NER/NEL ni se lanza entrenamiento.

## Revisión de arquitectura

La revisión detectó que la preparación de entrenamiento descartaba anotaciones, que la
selección anterior redondeaba por ventana y que no había un consumidor CLI para estadísticas
multilingües sobre varios Parquet. Esta funcionalidad introduce un consumidor separado de
los contratos canónicos: `corpus.py` para lectura/segmentación, `signals.py` para evidencia,
`selection.py` para selección proporcional pura, `reporting.py` para persistencia y
`workflow.py` para orquestación. El driver solo construye YAML y ejecuta la CLI.
El selector compartido de unidades se hizo público como `select_budget_units`.

Verificación: `python verification/verify_inspection.py` con fixtures locales sin descargas.

## Validación de esta entrega

- `verification/verify_inspection.py`: 263 comprobaciones pasan, con corpus anotados y
  sin anotar, múltiples corpus, todos los idiomas ISO admitidos, cuotas factibles e
  imposibles, backfill/underfill, CLI/script y entropía calculada por un modelo local.
- `verification/verify_cpt.py`: 186 pasan tras la refactorización del selector compartido.
- `verification/verify_layering.py`: 161 pasan; Ruff y formato pasan.
- Suite anterior: 14 pruebas pasan.

Se usaron fixtures sintéticos: la imagen recibida describe el formato, pero no contiene
el archivo `train_full.parquet`. La regresión completa heredada conserva el bloqueo de
acceso a los tokenizadores externos descrito en `STATUS.md`; estas pruebas nuevas son offline.
