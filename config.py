"""
Configuración central del traductor en cadena español -> latín / coreano / hindi.

Un solo lugar con los modelos, los idiomas y los parámetros de generación,
para que traductor.py, cadena.py y evaluar.py no se desincronicen.
"""

from pathlib import Path

import torch

# =========================================================
# RUTAS
# =========================================================

RAIZ = Path(__file__).resolve().parent
RUTA_SALIDAS = RAIZ / "salidas"
RUTA_FRASES_PRUEBA = RAIZ / "frases_prueba.txt"
RUTA_RESULTADOS_CSV = RUTA_SALIDAS / "resultados_cadena.csv"
RUTA_RESUMEN_CSV = RUTA_SALIDAS / "resumen_por_paso.csv"
RUTA_GRAFICA = RUTA_SALIDAS / "similitud_por_paso.png"

# =========================================================
# MODELOS (los dos son Transformers de Hugging Face)
# =========================================================

# Traductor: TranslateGemma 4B (Google, enero 2026), Transformer basado en
# Gemma 3 y afinado solo para traducir; soporta los 4 idiomas, LATÍN incluido.
# NLLB-200 y M2M-100 (los traductores multilingües más usados) no traen latín,
# y los Helsinki-NLP/opus-mt no tienen pares español-latín ni español-hindi.
# Primero se probó MADLAD-400 3B (google/madlad400-3b-mt), que sí dice
# soportar latín, pero en la práctica inventaba frases ("Incolae Portuenses
# appellantur" para "Los soldados defendieron la puerta de la ciudad").
# Con un solo modelo para todos los pares, cada salida pasa directo al
# siguiente idioma sin pivotear por inglés.
#
# Es la copia de Infomaniak-AI del modelo oficial (google/translategemma-4b-it),
# mismos pesos pero sin tener que pedir acceso en Hugging Face.
MODELO_TRADUCCION = "Infomaniak-AI/vllm-translategemma-4b-it"

# Verificador de significado: LaBSE (Google), Transformer tipo BERT que
# convierte una frase de cualquiera de 109 idiomas (latín incluido) en un
# vector; frases que significan lo mismo quedan cerca aunque estén en
# idiomas distintos. Con él se mide si cada salida de la cadena sigue
# significando lo mismo que la frase original en español.
MODELO_SIMILITUD = "sentence-transformers/LaBSE"

# =========================================================
# IDIOMAS (códigos ISO 639-1, los que espera TranslateGemma)
# =========================================================

IDIOMAS = {
    "es": "español",
    "la": "latín",
    "ko": "coreano",
    "hi": "hindi",
}
IDIOMA_ORIGEN = "es"
IDIOMAS_DESTINO = ["la", "ko", "hi"]

# Orden por defecto de la cadena: es -> la -> ko -> hi -> la -> ko -> hi ...
CADENA_POR_DEFECTO = ["la", "ko", "hi"] * 2

# =========================================================
# GENERACIÓN Y CONTROL DEL SIGNIFICADO
# =========================================================

# Cada paso genera varias traducciones candidatas con beam search y se queda
# con la que LaBSE dice que más se parece a la frase ORIGINAL en español (no
# solo a la del paso anterior). Así el error de un paso no se va acumulando:
# cada paso se "re-ancla" al significado original.
NUM_CANDIDATOS = 4
MAX_TOKENS_NUEVOS = 128

# Similitud coseno de LaBSE por debajo de la cual se considera que el
# significado se perdió. Paráfrasis correctas entre idiomas suelen quedar
# arriba de 0.80.
UMBRAL_SIMILITUD = 0.75

# Si el mejor candidato queda debajo del umbral, se generan además candidatos
# traduciendo desde la frase original en español y se elige el mejor de
# todos. El paso queda marcado como "corregido" en la salida, para que se
# vea cuándo pasó. Con False la cadena es pura (solo traduce la salida
# anterior) y solo se avisa de la pérdida.
CORREGIR_DERIVA = True

# =========================================================
# DATOS DE OPUS (https://opus.nlpl.eu) — corpus paralelos: frases con su traducción
# =========================================================
# (corpus, versión). Las URLs se arman en opus.py con el formato de descarga de OPUS.
#
# - Tatoeba: frases cortas traducidas por voluntarios; es la fuente más limpia
#   de español-latín (4,685 pares). Se reservan PRUEBA_LATIN frases para medir
#   y el resto se usa para afinar.
# - WikiMatrix: pares sacados de Wikipedia alineando frases automáticamente;
#   trae muchos pares mal emparejados, por eso se filtra con LaBSE.
# - TED2020: charlas TED traducidas por voluntarios. Tatoeba casi no tiene
#   español-hindi (34 pares), así que para medir coreano e hindi se usa TED2020.
RUTA_DATOS_OPUS = RAIZ / "datos_opus"
CORPUS_LATIN_LIMPIO = ("Tatoeba", "v2026-07-08")
CORPUS_LATIN_EXTRA = ("WikiMatrix", "v1")
CORPUS_PRUEBA_KO_HI = ("TED2020", "v1")

PRUEBA_LATIN = 300            # frases de Tatoeba que nunca se usan para entrenar
PRUEBA_KO_HI = 200            # frases de TED2020 por idioma
PALABRAS_MIN, PALABRAS_MAX = 3, 25
UMBRAL_LABSE_WIKIMATRIX = 0.80  # pares de WikiMatrix por debajo se descartan
UMBRAL_LABSE_REFERENCIAS = 0.75  # referencias de TED2020 cortadas o mal alineadas
SEMILLA = 42

# =========================================================
# FINE-TUNING DEL LATÍN CON LoRA
# =========================================================
# LoRA congela los 4,300 millones de parámetros del modelo y entrena solo dos
# matrices pequeñas (de rango LORA_RANGO) junto a las proyecciones de atención
# (q, k, v, o). Así cabe en la GPU gratuita de Colab y no se "olvida" lo que el
# modelo ya sabía de coreano e hindi. Se entrena en las dos direcciones
# (es->la y la->es) porque la cadena también lee latín.
RUTA_ADAPTADOR_LATIN = RAIZ / "adaptador_latin"
USAR_ADAPTADOR_LATIN = True     # si existe el adaptador, se usa en los pasos con latín
LORA_RANGO = 16
LORA_ALFA = 32
LORA_DROPOUT = 0.05
# En una T4 cada paso tarda ~13 s (bfloat16 es lento en esa GPU): 1,500 pares
# x 2 direcciones = 375 pasos ~ 80 min. Con 4,000 pares eran ~3.5 h, demasiado
# para Colab gratuito.
MAX_PARES_ENTRENAMIENTO = 1500
EPOCAS_AFINADO = 1
TAM_LOTE_AFINADO = 4
PASOS_ACUMULACION = 2           # lote efectivo = 8
TASA_APRENDIZAJE_LORA = 2e-4
GUARDAR_CADA = 200              # pasos; por si Colab se desconecta
TAM_LOTE_TRADUCCION = 16        # para evaluar con OPUS

DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# En GPU (Colab) bfloat16: la mitad de memoria (~9 GB, cabe en una T4 de 16 GB).
# En CPU float32: ~17 GB de RAM pero unas 3 veces más rápido que bfloat16.
TIPO_DATOS = torch.bfloat16 if DISPOSITIVO.type == "cuda" else torch.float32
