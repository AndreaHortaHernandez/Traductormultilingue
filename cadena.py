"""
Traducción en cadena: la frase en español pasa a un idioma, esa salida pasa
a otro, esa salida a otro, y así. Ejemplo con la cadena por defecto:

    español -> latín -> coreano -> hindi -> latín -> coreano -> hindi

El problema de encadenar traducciones es que el error se acumula, como en
el juego del "teléfono descompuesto": si el paso 2 pierde un matiz, el paso 3
traduce algo que ya no es lo original. Para que la salida siempre signifique
lo mismo, en cada paso:

1. TranslateGemma genera varias traducciones candidatas de la salida anterior.
2. LaBSE compara cada candidata con la frase ORIGINAL en español y se queda
   con la más parecida en significado (re-anclaje).
3. Si aun así la mejor queda debajo de config.UMBRAL_SIMILITUD y
   config.CORREGIR_DERIVA está activo, también se traduce desde el español
   original y se elige la mejor de todas; el paso queda marcado "corregido".

Ida y vuelta: LaBSE no distingue "quiero" de "quisiera / me gustaría", así
que con config.PESO_IDA_Y_VUELTA > 0 cada candidata se traduce además de
vuelta al español y se premia la que regresa con las mismas palabras que la
original (chrF). Así no se cuela, p. ej., la forma cortés del coreano.
"""

import unicodedata
from dataclasses import dataclass

from sacrebleu.metrics import CHRF

import config
from modelos import Traductor, Verificador

_CHRF = CHRF()


@dataclass
class Paso:
    numero: int
    idioma_entrada: str
    idioma_salida: str
    texto: str
    similitud: float          # con la frase original en español (LaBSE)
    regreso_es: str = ""      # la salida traducida de vuelta al español, para leerla
    corregido: bool = False   # True si se tuvo que traducir desde el original


def traducir_en_cadena(
    frase: str,
    cadena: list[str],
    traductor: Traductor,
    verificador: Verificador,
    con_regreso: bool = True,
    num_candidatos: int = config.NUM_CANDIDATOS,
    corregir_deriva: bool = config.CORREGIR_DERIVA,
    peso_ida_y_vuelta: float | None = None,
    umbral: float | None = None,
) -> list[Paso]:
    """
    Con num_candidatos=1 y corregir_deriva=False es la cadena "ingenua" (cada
    paso solo traduce la salida anterior, sin verificar); evaluar.py la usa
    como línea base para comparar.

    peso_ida_y_vuelta y umbral valen por defecto lo de config; la interfaz los
    cambia por consulta sin tocar config.
    """
    peso = config.PESO_IDA_Y_VUELTA if peso_ida_y_vuelta is None else peso_ida_y_vuelta
    umbral = config.UMBRAL_SIMILITUD if umbral is None else umbral
    # Acentos "descompuestos" (e + ´, común al pegar desde PDFs o Mac) se ven
    # igual pero son otros caracteres; en NFC "é" es un solo carácter y el
    # chrF de ida y vuelta no los cuenta como diferencias.
    frase = unicodedata.normalize("NFC", frase).strip()
    pasos = []
    texto_actual, idioma_actual = frase, config.IDIOMA_ORIGEN
    # Con un solo candidato no hay nada que elegir: la cadena ingenua no cambia.
    ida_y_vuelta = num_candidatos > 1 and peso > 0

    def puntuar(candidatos: list[str], idioma: str):
        """Similitud LaBSE con el original, puntaje para elegir y, si se usa
        ida y vuelta, el regreso al español de cada candidata."""
        similitudes = verificador.similitudes(frase, candidatos)
        if not ida_y_vuelta:
            return similitudes, similitudes, None
        regresos = traductor.traducir_lote(candidatos, idioma, config.IDIOMA_ORIGEN)
        puntajes = [
            s + peso * _CHRF.sentence_score(r, [frase]).score / 100
            for s, r in zip(similitudes, regresos)
        ]
        return similitudes, puntajes, regresos

    for numero, idioma_destino in enumerate(cadena, start=1):
        candidatos = traductor.traducir(texto_actual, idioma_actual, idioma_destino, num_candidatos)
        similitudes, puntajes, regresos = puntuar(candidatos, idioma_destino)
        mejor = max(range(len(candidatos)), key=puntajes.__getitem__)
        corregido = False

        if similitudes[mejor] < umbral and corregir_deriva:
            desde_original = traductor.traducir(
                frase, config.IDIOMA_ORIGEN, idioma_destino, num_candidatos,
            )
            nuevos = [c for c in desde_original if c not in candidatos]
            if nuevos:
                n_similitudes, n_puntajes, n_regresos = puntuar(nuevos, idioma_destino)
                if max(n_similitudes) > similitudes[mejor]:
                    candidatos += nuevos
                    similitudes += n_similitudes
                    puntajes = puntajes + n_puntajes if ida_y_vuelta else similitudes
                    if regresos is not None:
                        regresos += n_regresos
                    mejor = max(range(len(candidatos)), key=puntajes.__getitem__)
                    corregido = True

        paso = Paso(
            numero=numero,
            idioma_entrada=idioma_actual,
            idioma_salida=idioma_destino,
            texto=candidatos[mejor],
            similitud=similitudes[mejor],
            corregido=corregido,
        )
        if regresos is not None:
            paso.regreso_es = regresos[mejor]
        elif con_regreso:
            paso.regreso_es = traductor.traducir(paso.texto, idioma_destino, config.IDIOMA_ORIGEN)[0]
        pasos.append(paso)

        texto_actual, idioma_actual = paso.texto, idioma_destino

    return pasos


def imprimir_pasos(frase: str, pasos: list[Paso]) -> None:
    print(f"\nOriginal ({config.IDIOMAS[config.IDIOMA_ORIGEN]}): {frase}")
    for paso in pasos:
        aviso = ""
        if paso.corregido:
            aviso = "  [corregido: se tradujo desde el original]"
        elif paso.similitud < config.UMBRAL_SIMILITUD:
            aviso = "  [ojo: el significado se alejó]"
        print(
            f"\n{paso.numero}. {config.IDIOMAS[paso.idioma_entrada]} -> "
            f"{config.IDIOMAS[paso.idioma_salida]} (similitud {paso.similitud:.3f}){aviso}"
        )
        print(f"   {paso.texto}")
        if paso.regreso_es:
            print(f"   (de vuelta al español: {paso.regreso_es})")
