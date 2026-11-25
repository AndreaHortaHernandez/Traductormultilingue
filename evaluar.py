"""
Evaluación: corre la cadena sobre las frases de frases_prueba.txt de dos
formas y compara cuánto se conserva el significado en cada paso.

- "ingenua": cada paso traduce la salida anterior con la traducción más
  probable del modelo, sin verificar nada (teléfono descompuesto).
- "re-anclada": la del proyecto (varios candidatos + LaBSE + corrección).

Métricas por paso (promedio sobre las frases):
- similitud LaBSE con la frase original (de 0 a 1, más alto = mismo significado)
- % de salidas arriba de config.UMBRAL_SIMILITUD
- chrF de la salida traducida de vuelta al español contra la original
  (coincidencia de caracteres, de 0 a 100; independiente de LaBSE)
- % de pasos corregidos (solo re-anclada)
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sacrebleu.metrics import CHRF

import config
from cadena import traducir_en_cadena
from modelos import Traductor, Verificador

MODOS = {
    "ingenua": {"num_candidatos": 1, "corregir_deriva": False},
    "re-anclada": {"num_candidatos": config.NUM_CANDIDATOS, "corregir_deriva": config.CORREGIR_DERIVA},
}


def leer_frases(ruta: Path = config.RUTA_FRASES_PRUEBA) -> list[str]:
    lineas = Path(ruta).read_text(encoding="utf-8").splitlines()
    return [linea.strip() for linea in lineas if linea.strip()]


def _rutas_salida(ruta_frases: Path) -> tuple[Path, Path, Path]:
    """Con frases_prueba.txt se usan los nombres de siempre; con otro archivo
    (p. ej. frases_complejas.txt) se agrega un sufijo para no pisarlos."""
    rutas = (config.RUTA_RESULTADOS_CSV, config.RUTA_RESUMEN_CSV, config.RUTA_GRAFICA)
    if ruta_frases.resolve() == config.RUTA_FRASES_PRUEBA.resolve():
        return rutas
    sufijo = "_" + ruta_frases.stem.removeprefix("frases_")
    return tuple(r.with_name(r.stem + sufijo + r.suffix) for r in rutas)


def evaluar(
    cadena: list[str] | None = None,
    limite: int | None = None,
    traductor: Traductor | None = None,
    verificador: Verificador | None = None,
    ruta_frases: str | Path = config.RUTA_FRASES_PRUEBA,
) -> pd.DataFrame:
    """Si ya hay modelos cargados (p. ej. en el notebook de Colab) se pasan
    aquí para no cargarlos dos veces en la GPU. Con ruta_frases se evalúa
    otro archivo (relativo a la carpeta del proyecto, p. ej. frases_complejas.txt)."""
    cadena = cadena or config.CADENA_POR_DEFECTO
    ruta_frases = Path(ruta_frases)
    if not ruta_frases.is_absolute():
        ruta_frases = config.RAIZ / ruta_frases
    frases = leer_frases(ruta_frases)[:limite]
    ruta_resultados, ruta_resumen, ruta_grafica = _rutas_salida(ruta_frases)
    config.RUTA_SALIDAS.mkdir(parents=True, exist_ok=True)

    traductor = traductor or Traductor()
    verificador = verificador or Verificador()
    chrf = CHRF()
    filas = []

    for modo, parametros in MODOS.items():
        for i, frase in enumerate(frases, start=1):
            inicio = time.time()
            pasos = traducir_en_cadena(frase, cadena, traductor, verificador, **parametros)
            print(f"[{modo}] {i}/{len(frases)} ({time.time() - inicio:.0f}s): {frase}")
            for paso in pasos:
                filas.append({
                    "modo": modo,
                    "frase": frase,
                    "paso": paso.numero,
                    "idioma": config.IDIOMAS[paso.idioma_salida],
                    "salida": paso.texto,
                    "regreso_es": paso.regreso_es,
                    "similitud": paso.similitud,
                    "chrf_regreso": chrf.sentence_score(paso.regreso_es, [frase]).score,
                    "corregido": paso.corregido,
                })

    resultados = pd.DataFrame(filas)
    resultados.to_csv(ruta_resultados, index=False, encoding="utf-8-sig")

    resumen = (
        resultados.assign(arriba_umbral=resultados["similitud"] >= config.UMBRAL_SIMILITUD)
        .groupby(["modo", "paso", "idioma"], sort=False)
        .agg(
            similitud_media=("similitud", "mean"),
            similitud_minima=("similitud", "min"),
            pct_arriba_umbral=("arriba_umbral", "mean"),
            chrf_regreso=("chrf_regreso", "mean"),
            pct_corregidos=("corregido", "mean"),
        )
        .reset_index()
    )
    resumen[["pct_arriba_umbral", "pct_corregidos"]] *= 100
    resumen.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")

    print("\n=== Resumen por paso ===")
    print(resumen.to_string(index=False, float_format="{:.3f}".format))
    _graficar(resumen, cadena, ruta_grafica)
    print(f"\nResultados en '{ruta_resultados}', '{ruta_resumen}' y '{ruta_grafica}'")
    return resumen


def _graficar(resumen: pd.DataFrame, cadena: list[str], ruta: Path = config.RUTA_GRAFICA) -> None:
    fig, eje = plt.subplots(figsize=(8, 4.5))
    for modo, datos in resumen.groupby("modo", sort=False):
        eje.plot(datos["paso"], datos["similitud_media"], marker="o", label=f"Cadena {modo}")
    eje.axhline(config.UMBRAL_SIMILITUD, color="gray", linestyle="--", label="Umbral")
    eje.set_xticks(range(1, len(cadena) + 1))
    eje.set_xticklabels([f"{i}\n{config.IDIOMAS[c]}" for i, c in enumerate(cadena, start=1)])
    eje.set_ylim(0, 1)
    eje.set_xlabel("Paso de la cadena")
    eje.set_ylabel("Similitud LaBSE con el original")
    eje.set_title("¿La frase sigue significando lo mismo?")
    eje.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    evaluar()
