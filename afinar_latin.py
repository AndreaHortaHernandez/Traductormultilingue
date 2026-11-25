"""
Fine-tuning de TranslateGemma en latín con los pares de OPUS (LoRA).

En vez de reentrenar los 4,300 millones de parámetros, LoRA congela el modelo
y agrega, junto a cada proyección de atención (q, k, v, o) del modelo de
lenguaje, dos matrices pequeñas A (r x d) y B (d x r) cuyo producto se suma
a la original: W' = W + B·A·(alfa/r). Solo A y B se entrenan (~0.3 % de los
parámetros), así que cabe en la GPU gratuita de Colab y lo que el modelo ya
sabía de los demás idiomas no se toca.

Cada ejemplo es la instrucción de traducción + la traducción correcta. La
pérdida (entropía cruzada) se calcula solo sobre los tokens de la traducción:
el modelo aprende a responder, no a repetir la instrucción.

Dos versiones:
- afinar():    v1, los primeros config.MAX_PARES_ENTRENAMIENTO pares, desde cero.
- afinar_v2(): parte del v1 y sigue solo con frases largas que el v1 no vio
               (config.PALABRAS_V2), en adaptador_latin_v2/. El v1 no se toca.

Se guarda cada config.GUARDAR_CADA pasos; si Colab se desconecta, volver a
correrlo continúa desde el último guardado.
"""

import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model

import config
from modelos import cargar_modelo_base, cargar_tokenizer, instruccion
from opus import leer_tsv

RUTA_ESTADO = config.RUTA_ADAPTADOR_LATIN / "estado.json"  # el del v1 (lo lee el notebook)


def pares_v1() -> list[tuple[str, str]]:
    return leer_tsv("entrenamiento_es-la.tsv")[:config.MAX_PARES_ENTRENAMIENTO]


def pares_v2() -> list[tuple[str, str]]:
    """Frases que el v1 no vio, de PALABRAS_V2 palabras en los dos idiomas."""
    minimo, maximo = config.PALABRAS_V2
    resto = leer_tsv("entrenamiento_es-la.tsv")[config.MAX_PARES_ENTRENAMIENTO:]
    largos = [
        (es, la) for es, la in resto
        if minimo <= len(es.split()) <= maximo and minimo - 2 <= len(la.split()) <= maximo
    ]
    return largos[:config.MAX_PARES_V2]


def _ejemplos(tokenizer, pares, agrupar_por_largo: bool) -> list[dict]:
    ejemplos = []
    for es, la in pares:
        # Las dos direcciones: la cadena escribe latín y también lo lee.
        for texto, origen, destino, respuesta in ((es, "es", "la", la), (la, "la", "es", es)):
            prompt = tokenizer(instruccion(tokenizer, texto, origen, destino), add_special_tokens=False)["input_ids"]
            objetivo = tokenizer(respuesta + "<end_of_turn>", add_special_tokens=False)["input_ids"]
            ejemplos.append({
                "input_ids": prompt + objetivo,
                # -100 = "no calcular pérdida aquí" (la instrucción)
                "labels": [-100] * len(prompt) + objetivo,
            })
    azar = random.Random(config.SEMILLA)
    if not agrupar_por_largo:
        azar.shuffle(ejemplos)
        return ejemplos
    # Lotes de ejemplos de largo parecido (menos relleno = menos cómputo); se
    # barajan los lotes, no los ejemplos, para que el orden siga siendo al azar.
    ejemplos.sort(key=lambda e: len(e["input_ids"]))
    tam = config.TAM_LOTE_AFINADO
    lotes = [ejemplos[i:i + tam] for i in range(0, len(ejemplos), tam)]
    azar.shuffle(lotes)
    return [e for lote in lotes for e in lote]


def _lote(ejemplos: list[dict], pad_id: int) -> dict:
    largo = max(len(e["input_ids"]) for e in ejemplos)
    ids, etiquetas, mascara = [], [], []
    for e in ejemplos:
        relleno = largo - len(e["input_ids"])
        ids.append(e["input_ids"] + [pad_id] * relleno)
        etiquetas.append(e["labels"] + [-100] * relleno)
        mascara.append([1] * len(e["input_ids"]) + [0] * relleno)
    return {
        "input_ids": torch.tensor(ids, device=config.DISPOSITIVO),
        "labels": torch.tensor(etiquetas, device=config.DISPOSITIVO),
        "attention_mask": torch.tensor(mascara, device=config.DISPOSITIVO),
    }


def _guardar(modelo, ruta: Path, paso: int, total: int, perdidas: list[float], extra: dict) -> None:
    modelo.save_pretrained(ruta)
    (ruta / "estado.json").write_text(json.dumps(
        {"paso": paso, "total": total, "terminado": paso >= total, "perdidas": perdidas, **extra}, indent=1,
    ), encoding="utf-8")


def _entrenar(
    pares: list[tuple[str, str]],
    ruta: Path,
    tasa: float,
    adaptador_inicial: Path | None = None,
    agrupar_por_largo: bool = False,
) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("El fine-tuning necesita GPU: córrelo en Colab (Entorno de ejecución -> GPU).")

    ruta_estado = ruta / "estado.json"
    estado = json.loads(ruta_estado.read_text(encoding="utf-8")) if ruta_estado.exists() else None
    if estado and estado["terminado"]:
        print(f"El adaptador ya está entrenado ({estado['total']} pasos). "
              f"Para reentrenar, borra la carpeta '{ruta.name}'.")
        return

    tokenizer = cargar_tokenizer()
    ejemplos = _ejemplos(tokenizer, pares, agrupar_por_largo)
    pasos_por_epoca = math.ceil(len(ejemplos) / (config.TAM_LOTE_AFINADO * config.PASOS_ACUMULACION))
    total = pasos_por_epoca * config.EPOCAS_AFINADO
    print(f"{len(ejemplos)} ejemplos ({len(ejemplos) // 2} pares de OPUS x 2 direcciones), "
          f"{total} pasos de optimización.")

    modelo = cargar_modelo_base()
    modelo.config.use_cache = False
    modelo.gradient_checkpointing_enable()  # recalcula activaciones: menos memoria
    modelo.enable_input_require_grads()

    if estado:
        print(f"Continuando desde el paso {estado['paso']} de {total}.")
        modelo = PeftModel.from_pretrained(modelo, ruta, is_trainable=True)
    elif adaptador_inicial is not None:
        if not (adaptador_inicial / "adapter_config.json").exists():
            raise SystemExit(f"No encuentro el adaptador de partida en '{adaptador_inicial}'.")
        print(f"Se parte del adaptador '{adaptador_inicial.name}' (no se modifica).")
        modelo = PeftModel.from_pretrained(modelo, adaptador_inicial, is_trainable=True)
    else:
        modelo = get_peft_model(modelo, LoraConfig(
            r=config.LORA_RANGO,
            lora_alpha=config.LORA_ALFA,
            lora_dropout=config.LORA_DROPOUT,
            # Solo la atención del modelo de lenguaje (no la parte de visión).
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj)",
            task_type="CAUSAL_LM",
        ))
    # Los pesos LoRA en float32 para que el optimizador no pierda precisión.
    entrenables = [p for p in modelo.parameters() if p.requires_grad]
    for p in entrenables:
        p.data = p.data.float()
    modelo.print_trainable_parameters()
    paso_inicial = estado["paso"] if estado else 0
    perdidas = estado["perdidas"] if estado else []
    extra = {"pares": len(pares), "parte_de": adaptador_inicial.name if adaptador_inicial else None}

    optimizador = torch.optim.AdamW(entrenables, lr=tasa)
    # La tasa de aprendizaje baja linealmente hasta 0 al final (también al continuar).
    planificador = torch.optim.lr_scheduler.LambdaLR(
        optimizador, lambda p: 1 - (paso_inicial + p) / total,
    )

    modelo.train()
    tam = config.TAM_LOTE_AFINADO
    inicio, perdida_acumulada = time.time(), 0.0
    for paso in range(paso_inicial, total):
        for micro in range(config.PASOS_ACUMULACION):
            indice = ((paso * config.PASOS_ACUMULACION + micro) * tam) % len(ejemplos)
            lote = _lote(ejemplos[indice:indice + tam] or ejemplos[:tam], tokenizer.pad_token_id)
            perdida = modelo(**lote).loss / config.PASOS_ACUMULACION
            perdida.backward()
            perdida_acumulada += perdida.item()
        torch.nn.utils.clip_grad_norm_(entrenables, 1.0)
        optimizador.step()
        planificador.step()
        optimizador.zero_grad()

        if (paso + 1) % 20 == 0:
            perdidas.append(round(perdida_acumulada / 20, 4))
            transcurrido = time.time() - inicio
            restante = transcurrido / (paso + 1 - paso_inicial) * (total - paso - 1)
            print(f"Paso {paso + 1}/{total} - pérdida {perdidas[-1]:.4f} "
                  f"- faltan ~{restante / 60:.0f} min")
            perdida_acumulada = 0.0
        if (paso + 1) % config.GUARDAR_CADA == 0 or paso + 1 == total:
            _guardar(modelo, ruta, paso + 1, total, perdidas, extra)

    print(f"\nAdaptador guardado en '{ruta}'.")


def afinar() -> None:
    """v1: desde cero con los primeros MAX_PARES_ENTRENAMIENTO pares."""
    _entrenar(pares_v1(), config.RUTA_ADAPTADOR_LATIN, config.TASA_APRENDIZAJE_LORA)


def afinar_v2() -> None:
    """v2: parte del v1 y sigue con frases largas que el v1 no vio."""
    pares = pares_v2()
    minimo, maximo = config.PALABRAS_V2
    print(f"v2: {len(pares)} pares nuevos de {minimo}-{maximo} palabras, partiendo del v1.")
    _entrenar(
        pares, config.RUTA_ADAPTADOR_LATIN_V2, config.TASA_APRENDIZAJE_V2,
        adaptador_inicial=config.RUTA_ADAPTADOR_LATIN, agrupar_por_largo=True,
    )


if __name__ == "__main__":
    afinar()
