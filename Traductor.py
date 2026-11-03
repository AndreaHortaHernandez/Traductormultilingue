"""
Punto de entrada único del traductor en cadena.

    python Traductor.py traducir "El perro duerme bajo la mesa."
    python Traductor.py traducir "El perro duerme." --cadena la ko hi ko la
    python Traductor.py interactivo
    python Traductor.py evaluar [--limite 5]
    python Traductor.py opus          # descarga y prepara los datos de OPUS
    python Traductor.py afinar        # fine-tuning del latín con OPUS (necesita GPU)
    python Traductor.py evaluar_opus  # BLEU / chrF++ contra traducciones humanas
"""

import argparse

import config


def _cadena(argumento: list[str] | None) -> list[str]:
    cadena = argumento or config.CADENA_POR_DEFECTO
    invalidos = [c for c in cadena if c not in config.IDIOMAS_DESTINO]
    if invalidos:
        raise SystemExit(f"Idiomas no válidos: {invalidos}. Usa: {', '.join(config.IDIOMAS_DESTINO)}")
    return cadena


def main():
    parser = argparse.ArgumentParser(description="Traductor en cadena español -> latín / coreano / hindi.")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_traducir = sub.add_parser("traducir", help="Traduce una frase en cadena.")
    p_traducir.add_argument("frase")
    p_traducir.add_argument("--cadena", nargs="+", help="Orden de idiomas, ej. la ko hi la")

    p_interactivo = sub.add_parser("interactivo", help="Pide frases hasta que escribas 'salir'.")
    p_interactivo.add_argument("--cadena", nargs="+")

    p_evaluar = sub.add_parser("evaluar", help="Mide la conservación del significado con frases_prueba.txt.")
    p_evaluar.add_argument("--cadena", nargs="+")
    p_evaluar.add_argument("--limite", type=int, help="Usar solo las primeras N frases.")

    sub.add_parser("opus", help="Descarga y prepara los datos de OPUS.")
    sub.add_parser("afinar", help="Fine-tuning del latín con los datos de OPUS (GPU).")
    sub.add_parser("evaluar_opus", help="BLEU / chrF++ contra las traducciones humanas de OPUS.")

    args = parser.parse_args()

    if args.comando == "opus":
        import opus
        opus.preparar()
        return
    if args.comando == "afinar":
        import afinar_latin
        afinar_latin.afinar()
        return
    if args.comando == "evaluar_opus":
        import evaluar_opus
        evaluar_opus.evaluar_opus()
        return

    cadena = _cadena(args.cadena)

    if args.comando == "evaluar":
        import evaluar
        evaluar.evaluar(cadena, args.limite)
        return

    from cadena import imprimir_pasos, traducir_en_cadena
    from modelos import Traductor, Verificador

    traductor, verificador = Traductor(), Verificador()
    print("Cadena: " + " -> ".join(config.IDIOMAS[c] for c in [config.IDIOMA_ORIGEN, *cadena]))

    if args.comando == "traducir":
        imprimir_pasos(args.frase, traducir_en_cadena(args.frase, cadena, traductor, verificador))
        return

    while True:
        frase = input("\nFrase en español ('salir' para terminar): ").strip()
        if frase.lower() == "salir":
            break
        if frase:
            imprimir_pasos(frase, traducir_en_cadena(frase, cadena, traductor, verificador))


if __name__ == "__main__":
    main()
