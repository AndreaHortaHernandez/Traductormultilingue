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

Se guarda en adaptador_latin/ cada config.GUARDAR_CADA pasos; si Colab se
desconecta, volver a correrlo continúa desde el último guardado.
"""

import json
import math
import random
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model

import config
from modelos import cargar_modelo_base, cargar_tokenizer, instruccion
from opus import leer_tsv

RUTA_ESTADO = config.RUTA_ADAPTADOR_LATIN / "estado.json"


def _ejemplos(tokenizer) -> list[dict]:
    pares = leer_tsv("entrenamiento_es-la.tsv")[:config.MAX_PARES_ENTRENAMIENTO]
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
    random.Random(config.SEMILLA).shuffle(ejemplos)
    return ejemplos


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


def _guardar(modelo, paso: int, total: int, perdidas: list[float]) -> None:
    modelo.save_pretrained(config.RUTA_ADAPTADOR_LATIN)
    RUTA_ESTADO.write_text(json.dumps(
        {"paso": paso, "total": total, "terminado": paso >= total, "perdidas": perdidas}, indent=1,
    ))


def afinar() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("El fine-tuning necesita GPU: córrelo en Colab (Entorno de ejecución -> GPU).")

    estado = json.loads(RUTA_ESTADO.read_text()) if RUTA_ESTADO.exists() else None
    if estado and estado["terminado"]:
        print(f"El adaptador ya está entrenado ({estado['total']} pasos). "
              f"Para reentrenar, borra la carpeta '{config.RUTA_ADAPTADOR_LATIN.name}'.")
        return

    tokenizer = cargar_tokenizer()
    ejemplos = _ejemplos(tokenizer)
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
        modelo = PeftModel.from_pretrained(modelo, config.RUTA_ADAPTADOR_LATIN, is_trainable=True)
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

    optimizador = torch.optim.AdamW(entrenables, lr=config.TASA_APRENDIZAJE_LORA)
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
            _guardar(modelo, paso + 1, total, perdidas)

    print(f"\nAdaptador guardado en '{config.RUTA_ADAPTADOR_LATIN}'.")


if __name__ == "__main__":
    afinar()
