"""
Evaluación con los conjuntos de prueba de OPUS (frases con traducción humana).

Para cada idioma se traduce en las dos direcciones (español -> idioma e
idioma -> español) y se compara contra la traducción humana con:

- BLEU: qué tantas secuencias de 1 a 4 palabras coinciden con la referencia
  (0-100). Es la métrica clásica de traducción automática; castiga los
  sinónimos aunque estén bien.
- chrF++: coincidencia de caracteres y palabras (0-100). Más justa con
  idiomas de mucha flexión como el latín, y con coreano e hindi.

Ambas se tokenizan con el tokenizador de FLORES-200 (sirve igual para
alfabeto latino, hangul y devanagari). En latín se mide el modelo base y el
afinado con OPUS (si existe el adaptador), para ver si el fine-tuning ayudó.
"""

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

import config
from modelos import Traductor
from opus import leer_tsv

RUTA_METRICAS = config.RUTA_SALIDAS / "metricas_opus.csv"
RUTA_TRADUCCIONES = config.RUTA_SALIDAS / "traducciones_opus.csv"


def evaluar_opus(traductor: Traductor | None = None) -> pd.DataFrame:
    traductor = traductor or Traductor()
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    bleu, chrf = BLEU(tokenize="flores200"), CHRF(word_order=2)
    filas, traducciones = [], []

    for idioma in config.IDIOMAS_DESTINO:
        pares = leer_tsv(f"prueba_es-{idioma}.tsv")
        espanol, otro = [es for es, _ in pares], [x for _, x in pares]
        modelos = ["base", "afinado OPUS"] if idioma == "la" and traductor.con_adaptador else ["base"]

        for modelo in modelos:
            usar_adaptador = modelo != "base"
            for origen, destino, fuentes, referencias in (
                ("es", idioma, espanol, otro), (idioma, "es", otro, espanol),
            ):
                direccion = f"{config.IDIOMAS[origen]} -> {config.IDIOMAS[destino]}"
                print(f"[{modelo}] {direccion}: {len(fuentes)} frases...")
                hipotesis = traductor.traducir_lote(fuentes, origen, destino, usar_adaptador)
                filas.append({
                    "modelo": modelo,
                    "direccion": direccion,
                    "frases": len(fuentes),
                    "corpus": "Tatoeba" if idioma == "la" else "TED2020",
                    "BLEU": bleu.corpus_score(hipotesis, [referencias]).score,
                    "chrF++": chrf.corpus_score(hipotesis, [referencias]).score,
                })
                traducciones += [
                    {"modelo": modelo, "direccion": direccion, "entrada": f, "referencia": r, "traduccion": h}
                    for f, r, h in zip(fuentes, referencias, hipotesis)
                ]

    metricas = pd.DataFrame(filas)
    metricas.to_csv(RUTA_METRICAS, index=False, encoding="utf-8-sig")
    pd.DataFrame(traducciones).to_csv(RUTA_TRADUCCIONES, index=False, encoding="utf-8-sig")
    print("\n=== Métricas contra traducciones humanas de OPUS ===")
    print(metricas.to_string(index=False, float_format="{:.1f}".format))
    print(f"\nGuardado en '{RUTA_METRICAS}' y '{RUTA_TRADUCCIONES}'")
    return metricas


if __name__ == "__main__":
    evaluar_opus()
