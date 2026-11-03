# Traductor en cadena: español → latín / coreano / hindi

Se escribe una frase en español y se traduce **en cadena**: el español pasa a
uno de los tres idiomas, esa salida pasa a otro, esa salida a otro, y así.
La condición es que **la salida siempre siga significando lo mismo** que la
frase original.

```
español → latín → coreano → hindi → latín → coreano → hindi
```

## Idea general

### 1. Traducción con un Transformer: TranslateGemma

Se usa **TranslateGemma 4B** ([`google/translategemma-4b-it`](https://huggingface.co/google/translategemma-4b-it),
Google, enero 2026), un **Transformer de solo decoder** basado en Gemma 3
(4,300 millones de parámetros) y afinado exclusivamente para traducir entre
55 idiomas principales y cientos más, latín incluido.

- Recibe una instrucción con el idioma de origen y de destino más el texto
  ("You are a professional Spanish (es) to Latin (la) translator... ").
- Genera la traducción **token por token**. En cada token, la
  **auto-atención** con varias cabezas "mira" toda la frase de entrada y lo
  que ya lleva traducido, para decidir la siguiente palabra.
- Con **beam search** se generan varias traducciones candidatas a la vez
  (las 4 secuencias más probables), no solo una.

**¿Por qué este modelo?** El latín es el idioma difícil:

| Modelo | ¿Latín? | Resultado |
|---|---|---|
| NLLB-200 (Meta), M2M-100 | No lo traen | descartados |
| Helsinki-NLP opus-mt | No hay pares es→la ni es→hi | descartados |
| MADLAD-400 3B (Google) | Sí, en papel | inventaba frases: *"Los soldados defendieron la puerta de la ciudad"* → *"Incolae Portuenses appellantur"* ("sus habitantes se llaman portuenses") |
| **TranslateGemma 4B** | **Sí** | *"Milites portas urbis defenderunt"* ✔ |

Con un solo modelo para los 4 idiomas, cada salida pasa directo al siguiente
idioma sin pivotear por inglés. Se usa la copia
[`Infomaniak-AI/vllm-translategemma-4b-it`](https://huggingface.co/Infomaniak-AI/vllm-translategemma-4b-it)
(mismos pesos, sin tener que pedir acceso en Hugging Face).

### 2. Verificar que el significado no cambie: LaBSE

El problema de encadenar traducciones es el **teléfono descompuesto**: si un
paso pierde un matiz, el siguiente traduce algo que ya no es lo original, y el
error se acumula.

Para medirlo se usa [`sentence-transformers/LaBSE`](https://huggingface.co/sentence-transformers/LaBSE),
un **Transformer tipo BERT** (solo encoder) que convierte una frase de
cualquiera de 109 idiomas, latín incluido, en un vector de 768 números. Las
frases que significan lo mismo quedan cerca aunque estén en idiomas distintos,
así que la **similitud coseno** entre la frase original en español y cada
salida dice qué tanto se conservó el significado (1 = idéntico).

### 3. Re-anclaje en cada paso

En cada paso de la cadena (`cadena.py`):

1. TranslateGemma genera **4 traducciones candidatas** de la salida anterior (beam search).
2. LaBSE compara cada candidata con la frase **original en español**, no solo
   con la del paso anterior, y se elige la más parecida. Así cada paso se
   re-ancla al significado original y el error no se acumula.
3. Si aun así la mejor queda debajo del umbral (`UMBRAL_SIMILITUD = 0.75`),
   también se traduce desde el español original y se elige la mejor de todas;
   el paso se marca como **corregido** en la salida.

Cada salida además se traduce de vuelta al español, para que se pueda leer
qué dice.

### 4. Datos de OPUS: afinar el latín y medir

[OPUS](https://opus.nlpl.eu) no tiene modelos: es una colección de **corpus
paralelos** (la misma frase en dos idiomas, alineada). Para nuestros pares:

| Par | Pares en OPUS | Qué se usa (`opus.py`) |
|---|---|---|
| español–latín | ~732 mil | **Tatoeba** (frases cortas traducidas por personas): 300 para prueba, el resto para entrenar. **WikiMatrix** (alineado automático de Wikipedia) filtrado: solo pares con similitud LaBSE ≥ 0.80 y las mismas cifras → 7,038 pares de entrenamiento |
| español–coreano | ~51 millones | **TED2020**: 200 frases de prueba (filtradas con LaBSE para quitar referencias rotas) |
| español–hindi | ~30 millones | **TED2020**: 200 frases de prueba (Tatoeba solo tiene 34) |

Ninguna frase de prueba (ni sus otras traducciones) aparece en el entrenamiento.

**Fine-tuning con LoRA** (`afinar_latin.py`): el latín es el idioma más débil,
así que se afina TranslateGemma con los pares de OPUS. LoRA congela los 4,300
millones de parámetros y entrena solo dos matrices pequeñas junto a cada
proyección de atención (q, k, v, o): `W' = W + B·A·(α/r)`, con r = 16 (~0.3 %
de los parámetros). Se entrena en las dos direcciones (es→la, la→es) con
entropía cruzada solo sobre los tokens de la traducción. El adaptador se
activa únicamente en los pasos con latín; coreano e hindi usan el modelo
original.

**Evaluación con OPUS** (`evaluar_opus.py`): se traducen las frases de prueba
en las dos direcciones y se comparan contra la traducción humana con **BLEU**
(coincidencia de secuencias de 1-4 palabras) y **chrF++** (coincidencia de
caracteres y palabras, más justa con el latín, el coreano y el hindi). En
latín se compara el modelo base contra el afinado.

## Archivos

| Archivo | Descripción |
|---|---|
| `config.py` | Modelos, idiomas, orden de la cadena, número de candidatos y umbral. |
| `modelos.py` | Carga TranslateGemma (traductor) y LaBSE (verificador de significado). |
| `cadena.py` | La traducción en cadena con re-anclaje y corrección. |
| `opus.py` | Descarga los corpus de OPUS y arma los conjuntos de prueba y de entrenamiento. |
| `afinar_latin.py` | Fine-tuning del latín con LoRA (necesita GPU). |
| `evaluar_opus.py` | BLEU / chrF++ contra las traducciones humanas de OPUS. |
| `evaluar.py` | Compara la cadena ingenua contra la re-anclada sobre `frases_prueba.txt`. |
| `datos_opus/` | Conjuntos de prueba y entrenamiento ya preparados (TSV). |
| `Traductor.py` | Punto de entrada único (`traducir` / `interactivo` / `evaluar`). |
| `frases_prueba.txt` | 20 frases en español para la evaluación. |
| `colab/Traductor_Colab.ipynb` | Notebook para Colab con GPU; trae todo el código adentro. |
| `salidas/` | Resultados de la evaluación (CSV y gráfica). |

## Cómo ejecutar (Google Colab, recomendado)

1. Abrir el notebook directo desde GitHub:
   **https://colab.research.google.com/github/AndreaHortaHernandez/Traductormultilingue/blob/main/colab/Traductor_Colab.ipynb**
2. **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU (T4)**.
3. **Entorno de ejecución → Ejecutar todas**. El notebook clona este repo,
   instala las librerías, afina el latín, traduce una frase de ejemplo y corre
   las dos evaluaciones. En total ~1.5-2 h en una T4 gratuita.
4. Quien entrene el adaptador de latín lo sube al repo con la sección 10
   (junto con las métricas de `salidas/`). Desde ahí, a los demás el notebook
   se lo trae al clonar y **se salta el entrenamiento**.
5. En la sección 7 se cambia la frase y el orden de idiomas para probar otras.

Para guardar cambios al notebook en el repo: **Archivo → Guardar una copia en GitHub**.

## Cómo ejecutar (local)

```powershell
python -m pip install -r requirements.txt

python Traductor.py traducir "El perro duerme bajo la mesa."
python Traductor.py traducir "El perro duerme bajo la mesa." --cadena hi la ko la
python Traductor.py interactivo
python Traductor.py evaluar              # tarda: 20 frases x 2 modos
python Traductor.py evaluar --limite 5   # versión rápida
```

La primera vez se descargan los modelos (~8.6 GB TranslateGemma + ~2 GB
LaBSE) a la caché de Hugging Face. Corre en CPU en float32 (necesita ~20 GB
de RAM libre; ~20 s por paso, ~4 min una cadena de 6 pasos); con GPU es mucho
más rápido.

## Evaluación

`evaluar.py` corre la cadena de 6 pasos sobre las 20 frases de dos formas:

- **Ingenua**: cada paso solo traduce la salida anterior con la traducción
  más probable, sin verificar.
- **Re-anclada**: la del proyecto.

Y reporta por paso: similitud LaBSE media y mínima con el original, % de
salidas arriba del umbral, **chrF** de la traducción de vuelta al español
contra la original (métrica de traducción automática independiente de LaBSE)
y % de pasos corregidos.

## Créditos de los datos

Corpus descargados de [OPUS](https://opus.nlpl.eu) (J. Tiedemann, *Parallel Data,
Tools and Interfaces in OPUS*, LREC 2012): **Tatoeba** (tatoeba.org, CC BY 2.0 FR),
**WikiMatrix** (Schwenk et al., 2019, CC BY-SA) y **TED2020** (Reimers y Gurevych,
2020, CC BY-NC-ND 4.0). Los conjuntos de `datos_opus/` son subconjuntos filtrados
de estos corpus y conservan sus licencias.
