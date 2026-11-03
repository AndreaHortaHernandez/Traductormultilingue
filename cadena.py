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
"""

from dataclasses import dataclass

import config
from modelos import Traductor, Verificador


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
) -> list[Paso]:
    """
    Con num_candidatos=1 y corregir_deriva=False es la cadena "ingenua" (cada
    paso solo traduce la salida anterior, sin verificar); evaluar.py la usa
    como línea base para comparar.
    """
    pasos = []
    texto_actual, idioma_actual = frase, config.IDIOMA_ORIGEN

    for numero, idioma_destino in enumerate(cadena, start=1):
        candidatos = traductor.traducir(texto_actual, idioma_actual, idioma_destino, num_candidatos)
        similitudes = verificador.similitudes(frase, candidatos)
        mejor = max(range(len(candidatos)), key=similitudes.__getitem__)
        corregido = False

        if similitudes[mejor] < config.UMBRAL_SIMILITUD and corregir_deriva:
            desde_original = traductor.traducir(
                frase, config.IDIOMA_ORIGEN, idioma_destino, num_candidatos,
            )
            nuevos = [c for c in desde_original if c not in candidatos]
            nuevas_similitudes = verificador.similitudes(frase, nuevos) if nuevos else []
            if nuevos and max(nuevas_similitudes) > similitudes[mejor]:
                candidatos += nuevos
                similitudes += nuevas_similitudes
                mejor = max(range(len(candidatos)), key=similitudes.__getitem__)
                corregido = True

        paso = Paso(
            numero=numero,
            idioma_entrada=idioma_actual,
            idioma_salida=idioma_destino,
            texto=candidatos[mejor],
            similitud=similitudes[mejor],
            corregido=corregido,
        )
        if con_regreso:
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
