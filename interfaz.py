"""
Interfaz web (Gradio) del traductor en cadena.

    python Traductor.py interfaz              # local, en http://127.0.0.1:7860
    python Traductor.py interfaz --compartir  # además, un enlace público temporal

En Colab (sección "Interfaz" del notebook) se llama con los modelos que ya
están cargados, para no ocupar la GPU dos veces:

    import interfaz
    interfaz.lanzar(traductor, verificador)
"""

import sys
import time
from functools import partial
import unicodedata

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # sin ventana: la figura se dibuja para la página
import matplotlib.pyplot as plt
import pandas as pd

import config
from cadena import traducir_en_cadena

COLUMNAS = ["Paso", "De → a", "Traducción", "De vuelta al español", "Similitud", "Estado"]


def _sin_acentos(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


# Se acepta el código o el nombre del idioma, con o sin acento ("latín" o "latin").
_ALIAS = {_sin_acentos(nombre): codigo for codigo, nombre in config.IDIOMAS.items()}
_ALIAS.update({codigo: codigo for codigo in config.IDIOMAS})


def leer_cadena(texto: str) -> list[str]:
    piezas = texto.replace(",", " ").replace("→", " ").replace("->", " ").split()
    cadena, invalidos = [], []
    for pieza in piezas:
        codigo = _ALIAS.get(_sin_acentos(pieza))
        if codigo in config.IDIOMAS_DESTINO:
            cadena.append(codigo)
        else:
            invalidos.append(pieza)
    if invalidos:
        raise gr.Error(
            f"No reconozco: {', '.join(invalidos)}. Usa "
            + ", ".join(f"{c} ({config.IDIOMAS[c]})" for c in config.IDIOMAS_DESTINO) + "."
        )
    if not cadena:
        raise gr.Error("Escribe al menos un idioma para la cadena.")
    return cadena


def texto_cadena(cadena: list[str]) -> str:
    """["la", "ko"] -> "latín → coreano" (lo que se muestra en el cuadro)."""
    return " → ".join(config.IDIOMAS[c] for c in cadena)


def _tabla(pasos, umbral: float) -> pd.DataFrame:
    filas = []
    for paso in pasos:
        if paso.corregido:
            estado = "corregido (desde el original)"
        elif paso.similitud < umbral:
            estado = "⚠ se alejó del significado"
        else:
            estado = "✔"
        filas.append([
            paso.numero,
            f"{config.IDIOMAS[paso.idioma_entrada]} → {config.IDIOMAS[paso.idioma_salida]}",
            paso.texto,
            paso.regreso_es,
            round(paso.similitud, 3),
            estado,
        ])
    return pd.DataFrame(filas, columns=COLUMNAS)


def _grafica(series: dict[str, list], cadena: list[str], umbral: float):
    figura, eje = plt.subplots(figsize=(7, 3.6))
    pasos = list(range(1, len(cadena) + 1))
    for nombre, (similitudes, estilo) in series.items():
        eje.plot(pasos, similitudes, estilo, label=nombre, linewidth=2, markersize=6)
    eje.axhline(umbral, color="gray", linestyle="--", linewidth=1, label=f"Umbral ({umbral:.2f})")
    eje.set_xticks(pasos, [f"{n}\n{config.IDIOMAS[c]}" for n, c in zip(pasos, cadena)])
    eje.set_ylim(0, 1.02)
    eje.set_ylabel("Similitud con el original (LaBSE)")
    eje.grid(axis="y", alpha=0.3)
    eje.legend(loc="lower left", fontsize=8)
    figura.tight_layout()
    plt.close(figura)
    return figura


def crear_interfaz(traductor, verificador) -> gr.Blocks:
    def traducir(frase, texto_cadena, num_candidatos, peso, umbral, corregir, comparar):
        frase = unicodedata.normalize("NFC", frase or "").strip()
        if not frase:
            raise gr.Error("Escribe una frase en español.")
        cadena = leer_cadena(texto_cadena)
        inicio = time.time()

        pasos = traducir_en_cadena(
            frase, cadena, traductor, verificador,
            num_candidatos=int(num_candidatos), corregir_deriva=corregir,
            peso_ida_y_vuelta=peso, umbral=umbral,
        )
        series = {"Re-anclada": ([p.similitud for p in pasos], "o-")}
        if comparar:
            ingenua = traducir_en_cadena(
                frase, cadena, traductor, verificador,
                con_regreso=False, num_candidatos=1, corregir_deriva=False,
            )
            series["Ingenua (sin verificar)"] = ([p.similitud for p in ingenua], "s--")

        final = pasos[-1]
        corregidos = sum(p.corregido for p in pasos)
        resumen = (
            f"**Ruta:** {' → '.join(config.IDIOMAS[c] for c in [config.IDIOMA_ORIGEN, *cadena])}\n\n"
            f"**Salida final ({config.IDIOMAS[final.idioma_salida]}):** {final.texto}\n\n"
            f"**De vuelta al español:** {final.regreso_es}\n\n"
            f"Similitud final **{final.similitud:.3f}** · mínima "
            f"**{min(p.similitud for p in pasos):.3f}** · pasos corregidos **{corregidos}** · "
            f"{time.time() - inicio:.0f} s"
        )
        if comparar:
            resumen += f"\n\nCadena ingenua: similitud final **{ingenua[-1].similitud:.3f}**."
        return resumen, _tabla(pasos, umbral), _grafica(series, cadena, umbral)

    try:
        ejemplos = config.RUTA_FRASES_PRUEBA.read_text(encoding="utf-8").splitlines()
        ejemplos = [[f] for f in ejemplos if f.strip()][:8]
    except OSError:
        ejemplos = []

    with gr.Blocks(title="Traductor en cadena") as demo:
        gr.Markdown(
            "# Traductor en cadena: español → latín / coreano / hindi\n"
            f"*Traductor: {getattr(config, 'NOMBRE_MODELO', config.MODELO_TRADUCCION)}*\n\n"
            "La frase pasa de un idioma a otro y en cada paso **LaBSE** verifica que siga "
            "significando lo mismo que el original. Acentos, ñ y ¿¡ funcionan normal."
        )
        with gr.Row():
            with gr.Column(scale=3):
                frase = gr.Textbox(
                    label="Frase en español", lines=2,
                    value="Quiero aprender a tocar la guitarra.",
                )
                cadena = gr.Textbox(
                    label="Idiomas a los que se traduce, en orden",
                    value=texto_cadena(config.CADENA_POR_DEFECTO),
                    info="Da clic en los idiomas en el orden que quieras; con uno solo se traduce directo a ese "
                         "idioma. También se puede escribir (la / latín, ko / coreano, hi / hindi).",
                )
                with gr.Row():
                    botones_idioma = {
                        codigo: gr.Button(f"+ {config.IDIOMAS[codigo]}", size="sm")
                        for codigo in config.IDIOMAS_DESTINO
                    }
                    deshacer = gr.Button("↶ Quitar el último", size="sm")
                    limpiar = gr.Button("Borrar", size="sm")
                with gr.Row():
                    gr.Markdown("Atajos:")
                    atajos = {
                        "Cadena completa (6 pasos)": config.CADENA_POR_DEFECTO,
                        "Una vuelta (3 pasos)": config.IDIOMAS_DESTINO,
                    }
                    botones_atajo = {nombre: gr.Button(nombre, size="sm") for nombre in atajos}
            with gr.Column(scale=2):
                with gr.Accordion("Parámetros", open=False):
                    num_candidatos = gr.Slider(
                        1, 6, value=config.NUM_CANDIDATOS, step=1, label="Candidatos por paso (beam search)",
                        info="Con 1 no hay nada que elegir: es la cadena ingenua.",
                    )
                    peso = gr.Slider(
                        0, 1, value=config.PESO_IDA_Y_VUELTA, step=0.1, label="Peso de ida y vuelta (chrF)",
                        info="0 = elegir solo por LaBSE. Más alto = más lento, pero respeta las palabras.",
                    )
                    umbral = gr.Slider(0.5, 0.95, value=config.UMBRAL_SIMILITUD, step=0.05, label="Umbral de similitud")
                    corregir = gr.Checkbox(value=config.CORREGIR_DERIVA, label="Corregir desde el original si baja del umbral")
                comparar = gr.Checkbox(value=False, label="Comparar con la cadena ingenua (tarda un poco más)")
                boton = gr.Button("Traducir", variant="primary")

        if ejemplos:
            gr.Examples(ejemplos, inputs=[frase], label="Frases de prueba")

        resumen = gr.Markdown()
        grafica = gr.Plot(label="Similitud por paso")
        tabla = gr.Dataframe(headers=COLUMNAS, label="Paso a paso", wrap=True, interactive=False)

        def agregar(actual, codigo):
            try:
                previa = leer_cadena(actual) if actual.strip() else []
            except gr.Error:
                previa = []  # si lo escrito no se entiende, se empieza de nuevo
            return texto_cadena([*previa, codigo])

        def quitar_ultimo(actual):
            try:
                return texto_cadena(leer_cadena(actual)[:-1])
            except gr.Error:
                return ""

        for codigo, boton_idioma in botones_idioma.items():
            boton_idioma.click(partial(agregar, codigo=codigo), cadena, cadena, api_name=f"agregar_{codigo}")
        deshacer.click(quitar_ultimo, cadena, cadena, api_name="quitar_ultimo")
        limpiar.click(lambda: "", None, cadena, api_name="borrar")
        for nombre, boton_atajo in botones_atajo.items():
            boton_atajo.click(partial(texto_cadena, atajos[nombre]), None, cadena, api_name=False)

        entradas = [frase, cadena, num_candidatos, peso, umbral, corregir, comparar]
        salidas = [resumen, tabla, grafica]
        # Un solo modelo en la GPU: las consultas se atienden de una en una.
        boton.click(traducir, entradas, salidas, concurrency_limit=1, api_name="traducir")
        frase.submit(traducir, entradas, salidas, concurrency_limit=1, api_name=False)
    return demo


def lanzar(traductor=None, verificador=None, compartir: bool | None = None, bloquear: bool = False):
    """
    Abre la interfaz. Si no se pasan los modelos, los carga. En Colab se crea
    un enlace público (*.gradio.live, dura 72 h) además de mostrarla en la celda.
    """
    if traductor is None or verificador is None:
        from modelos import Traductor, Verificador
        traductor = traductor or Traductor()
        verificador = verificador or Verificador()
    if compartir is None:
        compartir = "google.colab" in sys.modules
    demo = crear_interfaz(traductor, verificador)
    demo.queue().launch(share=compartir, prevent_thread_lock=not bloquear)
    return demo
