"""
Descarga y prepara los datos de OPUS (https://opus.nlpl.eu).

OPUS no tiene modelos: es una colección de corpus paralelos (la misma frase en
dos idiomas). Aquí se usa para dos cosas:

1. PRUEBA: frases con traducción humana para medir el traductor con BLEU y
   chrF++ (las métricas estándar de traducción automática), en vez de solo
   con frases inventadas.
2. ENTRENAMIENTO: pares español-latín para afinar el modelo en latín, el
   idioma en el que más falla.

Archivos que deja en datos_opus/ (TSV: español <tab> otro idioma):
    prueba_es-la.tsv, prueba_es-ko.tsv, prueba_es-hi.tsv, entrenamiento_es-la.tsv
"""

import csv
import io
import random
import re
import urllib.request
import zipfile

import config

_INVISIBLES = re.compile(r"[​‌‍﻿]")


def _url(corpus: str, version: str, par: str) -> str:
    return f"https://object.pouta.csc.fi/OPUS-{corpus}/{version}/moses/{par}.txt.zip"


def descargar_pares(corpus: str, version: str, idioma: str) -> list[tuple[str, str]]:
    """Descarga (o reutiliza) el zip de OPUS y regresa [(español, otro_idioma)]."""
    par = f"es-{idioma}"
    ruta_zip = config.RUTA_DATOS_OPUS / "zips" / f"{corpus}_{version}_{par}.zip"
    if not ruta_zip.exists():
        ruta_zip.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando {corpus} {par} de OPUS...")
        urllib.request.urlretrieve(_url(corpus, version, par), ruta_zip)

    with zipfile.ZipFile(ruta_zip) as zf:
        def leer(extension):
            with zf.open(f"{corpus}.{par}.{extension}") as archivo:
                return io.TextIOWrapper(archivo, encoding="utf-8").read().splitlines()
        espanol, otro = leer("es"), leer(idioma)

    pares = []
    for es, x in zip(espanol, otro):
        es, x = _INVISIBLES.sub("", es).strip(), _INVISIBLES.sub("", x).strip()
        if es and x:
            pares.append((es, x))
    return pares


def _largo_valido(frase: str) -> bool:
    return config.PALABRAS_MIN <= len(frase.split()) <= config.PALABRAS_MAX


def _mismos_numeros(a: str, b: str) -> bool:
    """WikiMatrix empareja frases parecidas de artículos distintos ("1600-1643,
    matemático italiano" con "1563-1643, mathematicus Anglicus"); LaBSE las ve
    parecidas, pero si las cifras no coinciden no son traducción una de otra."""
    return sorted(re.findall(r"\d+", a)) == sorted(re.findall(r"\d+", b))


def _filtrar_con_labse(pares, verificador, umbral):
    similitudes = verificador.similitudes_por_par([a for a, _ in pares], [b for _, b in pares])
    return [par for par, s in zip(pares, similitudes) if s >= umbral]


def _guardar(nombre: str, pares: list[tuple[str, str]]) -> None:
    ruta = config.RUTA_DATOS_OPUS / nombre
    with open(ruta, "w", newline="", encoding="utf-8") as archivo:
        csv.writer(archivo, delimiter="\t").writerows(pares)
    print(f"  {nombre}: {len(pares)} pares")


def leer_tsv(nombre: str) -> list[tuple[str, str]]:
    with open(config.RUTA_DATOS_OPUS / nombre, newline="", encoding="utf-8") as archivo:
        return [(fila[0], fila[1]) for fila in csv.reader(archivo, delimiter="\t")]


def preparar(verificador=None) -> None:
    """
    Arma los conjuntos de prueba y de entrenamiento. `verificador` (LaBSE) se
    usa para filtrar WikiMatrix; si no se pasa, se carga aquí.
    """
    config.RUTA_DATOS_OPUS.mkdir(parents=True, exist_ok=True)
    azar = random.Random(config.SEMILLA)

    # --- Latín: Tatoeba se parte en prueba / entrenamiento ---
    tatoeba = descargar_pares(*config.CORPUS_LATIN_LIMPIO, "la")
    # Una frase en español puede tener varias traducciones al latín: la prueba
    # se elige por frase en español y TODAS sus traducciones salen del
    # entrenamiento, para que el modelo nunca haya visto las frases de prueba.
    por_frase: dict[str, list[str]] = {}
    for es, la in tatoeba:
        por_frase.setdefault(es, []).append(la)
    candidatas = sorted(es for es in por_frase if _largo_valido(es))
    prueba_es = set(azar.sample(candidatas, config.PRUEBA_LATIN))
    prueba_la = [(es, por_frase[es][0]) for es in sorted(prueba_es)]
    latin_de_prueba = {la for es in prueba_es for la in por_frase[es]}
    entrenamiento = [
        (es, la) for es, la in tatoeba
        if es not in prueba_es and la not in latin_de_prueba
    ]

    # --- Latín extra: WikiMatrix filtrado con LaBSE ---
    wikimatrix = [
        (es, la) for es, la in descargar_pares(*config.CORPUS_LATIN_EXTRA, "la")
        if _largo_valido(es) and _largo_valido(la) and _mismos_numeros(es, la)
        and es not in prueba_es and la not in latin_de_prueba
    ]
    if verificador is None:
        from modelos import Verificador
        verificador = Verificador()
    print(f"Filtrando {len(wikimatrix)} pares de WikiMatrix con LaBSE "
          f"(umbral {config.UMBRAL_LABSE_WIKIMATRIX})...")
    buenos = _filtrar_con_labse(wikimatrix, verificador, config.UMBRAL_LABSE_WIKIMATRIX)
    print(f"  WikiMatrix: se quedan {len(buenos)} de {len(wikimatrix)}")
    entrenamiento = list(dict.fromkeys(entrenamiento + buenos))
    azar.shuffle(entrenamiento)

    # --- Coreano e hindi: solo prueba, de TED2020 ---
    # Algunas traducciones de TED están cortadas o repetidas; se descartan con
    # LaBSE para no castigar al traductor por errores de la referencia.
    pruebas = {"la": prueba_la}
    for idioma in ("ko", "hi"):
        ted = [
            par for par in descargar_pares(*config.CORPUS_PRUEBA_KO_HI, idioma)
            if _largo_valido(par[0])
        ]
        muestra = azar.sample(ted, config.PRUEBA_KO_HI * 3)
        limpios = _filtrar_con_labse(muestra, verificador, config.UMBRAL_LABSE_REFERENCIAS)
        pruebas[idioma] = sorted(limpios[:config.PRUEBA_KO_HI])

    print("Conjuntos guardados en datos_opus/:")
    for idioma, pares in pruebas.items():
        _guardar(f"prueba_es-{idioma}.tsv", pares)
    _guardar("entrenamiento_es-la.tsv", entrenamiento)


if __name__ == "__main__":
    preparar()
