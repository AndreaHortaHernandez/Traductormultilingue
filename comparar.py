"""
Compara configuraciones de la cadena re-anclada sobre las mismas frases, para
elegir la que mejor conserva el significado SIN volver a entrenar.

    python Traductor.py comparar [--frases frases_complejas.txt] [--limite 10]

Variantes (todas con la cadena por defecto y la ida y vuelta de config):
- actual:            lo de config.py (adaptador de latín, umbral 0.75, 4 candidatos)
- sin adaptador:     el latín con el TranslateGemma original. El adaptador se
                     entrenó con frases cortas de Tatoeba y podría empeorar las
                     frases complejas.
- umbral 0.85 + 6:   corrige desde el original más pasos dudosos y elige entre
                     más candidatos.
- sin adaptador + umbral 0.85 + 6: las dos cosas.
- adaptador v2:      el v2 (afinar_v2, frases largas), si existe adaptador_latin_v2/.
- v2 + umbral 0.85 + 6.

Para que la comparación sea justa, "% arriba de 0.75" se mide con el mismo
umbral en todas, aunque la variante use otro para decidir cuándo corregir.
"""

import time
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import CHRF

import config
from cadena import traducir_en_cadena
from evaluar import leer_frases

UMBRAL_REPORTE = 0.75

VARIANTES = {
    "actual": {"adaptador": True, "umbral": config.UMBRAL_SIMILITUD, "num_candidatos": config.NUM_CANDIDATOS},
    "sin adaptador": {"adaptador": False, "umbral": config.UMBRAL_SIMILITUD, "num_candidatos": config.NUM_CANDIDATOS},
    "umbral 0.85 + 6 cand.": {"adaptador": True, "umbral": 0.85, "num_candidatos": 6},
    "sin adaptador + umbral 0.85 + 6 cand.": {"adaptador": False, "umbral": 0.85, "num_candidatos": 6},
    "adaptador v2": {"adaptador": "v2", "umbral": config.UMBRAL_SIMILITUD, "num_candidatos": config.NUM_CANDIDATOS},
    "v2 + umbral 0.85 + 6 cand.": {"adaptador": "v2", "umbral": 0.85, "num_candidatos": 6},
}
VARIANTES_V1 = [n for n, v in VARIANTES.items() if v["adaptador"] != "v2"]
VARIANTES_V2 = ["actual", "adaptador v2", "sin adaptador"]
# Rama modelo-12b. "12B" y "12B + umbral..." son sin adaptador (así se corrieron
# la primera vez); las "+ adaptador" usan el de adaptador_latin_12b/.
VARIANTES.update({
    "12B": {"adaptador": False, "umbral": config.UMBRAL_SIMILITUD, "num_candidatos": config.NUM_CANDIDATOS},
    "12B + umbral 0.85 + 6 cand.": {"adaptador": False, "umbral": 0.85, "num_candidatos": 6},
    "12B + adaptador": {"adaptador": True, "umbral": config.UMBRAL_SIMILITUD, "num_candidatos": config.NUM_CANDIDATOS},
    "12B + adaptador + umbral 0.85 + 6 cand.": {"adaptador": True, "umbral": 0.85, "num_candidatos": 6},
})
VARIANTES_12B = ["12B", "12B + umbral 0.85 + 6 cand."]
VARIANTES_12B_ADAPTADOR = ["12B + adaptador + umbral 0.85 + 6 cand."]

RUTA_DETALLE = config.RUTA_SALIDAS / "comparacion_variantes.csv"
RUTA_RESUMEN = config.RUTA_SALIDAS / "comparacion_resumen.csv"


class _ConAdaptador:
    """Envuelve al traductor para que todos sus pasos usen un adaptador dado
    (False = el modelo original, o el nombre de uno extra como "v2"), sin
    recargar el modelo en la GPU."""

    def __init__(self, traductor, adaptador):
        self._t, self._a = traductor, adaptador

    def traducir(self, texto, origen, destino, num_candidatos=1):
        return self._t.traducir(texto, origen, destino, num_candidatos, usar_adaptador=self._a)

    def traducir_lote(self, textos, origen, destino):
        return self._t.traducir_lote(textos, origen, destino, usar_adaptador=self._a)


def comparar(
    traductor=None,
    verificador=None,
    ruta_frases: str | Path = "frases_complejas.txt",
    limite: int | None = None,
    variantes: list[str] | None = None,
) -> pd.DataFrame:
    if traductor is None or verificador is None:
        from modelos import Traductor, Verificador
        traductor = traductor or Traductor()
        verificador = verificador or Verificador()
    ruta_frases = Path(ruta_frases)
    if not ruta_frases.is_absolute():
        ruta_frases = config.RAIZ / ruta_frases
    frases = leer_frases(ruta_frases)[:limite]
    if variantes is None:
        variantes = VARIANTES_V1
        if config.RUTA_ADAPTADOR_LATIN_V2.exists() and "v2" not in getattr(traductor, "adaptadores_extra", []):
            traductor.agregar_adaptador("v2", config.RUTA_ADAPTADOR_LATIN_V2)
        if "v2" in getattr(traductor, "adaptadores_extra", []):
            variantes = VARIANTES_V1 + ["adaptador v2", "v2 + umbral 0.85 + 6 cand."]
    nombres = list(variantes)
    if any(VARIANTES[n]["adaptador"] == "v2" for n in nombres) and "v2" not in getattr(traductor, "adaptadores_extra", []):
        if not traductor.agregar_adaptador("v2", config.RUTA_ADAPTADOR_LATIN_V2):
            raise SystemExit("Las variantes 'v2' necesitan adaptador_latin_v2/ (entrénalo con afinar_v2).")
    cadena = config.CADENA_POR_DEFECTO
    chrf = CHRF()
    filas = []

    if not getattr(traductor, "con_adaptador", False):
        print("Aviso: no hay adaptador de latín cargado; 'con' y 'sin adaptador' darán lo mismo.")

    for nombre in nombres:
        v = VARIANTES[nombre]
        t = traductor if v["adaptador"] is True else _ConAdaptador(traductor, v["adaptador"])
        inicio_variante = time.time()
        for i, frase in enumerate(frases, start=1):
            inicio = time.time()
            pasos = traducir_en_cadena(
                frase, cadena, t, verificador,
                num_candidatos=v["num_candidatos"], umbral=v["umbral"],
            )
            segundos = time.time() - inicio
            print(f"[{nombre}] {i}/{len(frases)} ({segundos:.0f}s): {frase}")
            for paso in pasos:
                filas.append({
                    "variante": nombre,
                    "frase": frase,
                    "paso": paso.numero,
                    "idioma": config.IDIOMAS[paso.idioma_salida],
                    "salida": paso.texto,
                    "regreso_es": paso.regreso_es,
                    "similitud": paso.similitud,
                    "chrf_regreso": chrf.sentence_score(paso.regreso_es, [frase]).score,
                    "corregido": paso.corregido,
                    "segundos_frase": segundos,
                })
        print(f"[{nombre}] terminó en {(time.time() - inicio_variante) / 60:.1f} min\n")

    detalle = pd.DataFrame(filas)
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)
    detalle.to_csv(RUTA_DETALLE, index=False, encoding="utf-8-sig")
    resumen = _resumir(detalle, nombres, len(cadena))
    resumen.to_csv(RUTA_RESUMEN, index=False, encoding="utf-8-sig")
    _imprimir(detalle, resumen, nombres, len(cadena))
    return resumen


def _resumir(detalle: pd.DataFrame, nombres: list[str], ultimo: int) -> pd.DataFrame:
    filas = []
    for nombre in nombres:
        d = detalle[detalle["variante"] == nombre]
        final = d[d["paso"] == ultimo]
        latin = d[d["idioma"] == config.IDIOMAS["la"]]
        filas.append({
            "variante": nombre,
            "similitud_media": d["similitud"].mean(),
            "similitud_final": final["similitud"].mean(),
            "similitud_minima": d["similitud"].min(),
            "similitud_latin": latin["similitud"].mean(),
            "pct_arriba_0.75": 100 * (d["similitud"] >= UMBRAL_REPORTE).mean(),
            "chrf_regreso": d["chrf_regreso"].mean(),
            "chrf_regreso_final": final["chrf_regreso"].mean(),
            "pct_corregidos": 100 * d["corregido"].mean(),
            "seg_por_frase": d.groupby("frase")["segundos_frase"].first().mean(),
        })
    return pd.DataFrame(filas)


def _imprimir(detalle: pd.DataFrame, resumen: pd.DataFrame, nombres: list[str], ultimo: int) -> None:
    pd.set_option("display.width", 250, "display.max_columns", None, "display.max_colwidth", 70)
    print("=== Resumen por variante (todos los pasos) ===")
    print(resumen.to_string(index=False, float_format="{:.3f}".format))

    final = detalle[detalle["paso"] == ultimo]
    print(f"\n=== Similitud del último paso (paso {ultimo}) por frase ===")
    tabla = final.pivot_table(index="frase", columns="variante", values="similitud", sort=False)[nombres]
    print(tabla.to_string(float_format="{:.3f}".format))

    print(f"\n=== Cómo regresa al español el último paso ===")
    for frase, grupo in final.groupby("frase", sort=False):
        print(f"\n{frase}")
        for nombre in nombres:
            fila = grupo[grupo["variante"] == nombre].iloc[0]
            print(f"   {nombre:<40} {fila['similitud']:.3f}  {fila['regreso_es']}")
    print(f"\nGuardado en '{RUTA_RESUMEN}' y '{RUTA_DETALLE}'")
