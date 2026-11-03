"""
Los dos Transformers del proyecto:

- Traductor: TranslateGemma 4B (solo decoder, basado en Gemma 3). Recibe una
  instrucción "traduce de X a Y: <texto>" y genera la traducción token por
  token; en cada token, la auto-atención "mira" toda la frase de entrada y
  lo que ya lleva traducido. Si existe el adaptador LoRA afinado con OPUS
  (afinar_latin.py), se activa en los pasos que involucran latín.
- Verificador: LaBSE (solo encoder, tipo BERT). Convierte una frase en un
  vector de 768 números; la similitud coseno entre dos vectores dice qué tan
  parecido es su significado, aunque estén en idiomas distintos.
"""

from contextlib import nullcontext

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForImageTextToText, AutoTokenizer

import config


def cargar_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODELO_TRADUCCION)
    tokenizer.padding_side = "left"  # en generación por lotes el relleno va a la izquierda
    return tokenizer


def cargar_modelo_base():
    # TranslateGemma también traduce texto de imágenes, por eso la clase es
    # "ImageTextToText"; aquí solo se le pasa texto. device_map carga los
    # pesos directo al dispositivo, sin pasar una copia completa por la RAM
    # (en Colab gratuito la RAM es menor que la memoria de la GPU).
    modelo = AutoModelForImageTextToText.from_pretrained(
        config.MODELO_TRADUCCION, dtype=config.TIPO_DATOS, device_map=str(config.DISPOSITIVO),
    )
    modelo.generation_config.max_length = None  # se usa max_new_tokens
    return modelo


def instruccion(tokenizer, texto: str, idioma_origen: str, idioma_destino: str) -> str:
    """
    La instrucción completa que recibe el modelo, en el formato de su
    plantilla de chat ("You are a professional <origen> to <destino>
    translator... " + texto). Se usa igual al traducir y al afinar.
    """
    mensaje = f"<<<source>>>{idioma_origen}<<<target>>>{idioma_destino}<<<text>>>{texto}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": mensaje}], tokenize=False, add_generation_prompt=True,
    )


class Traductor:
    def __init__(self, usar_adaptador: bool = config.USAR_ADAPTADOR_LATIN):
        print(f"Cargando traductor '{config.MODELO_TRADUCCION}' en {config.DISPOSITIVO}...")
        self.tokenizer = cargar_tokenizer()
        self.modelo = cargar_modelo_base()
        self.con_adaptador = False
        if usar_adaptador and (config.RUTA_ADAPTADOR_LATIN / "adapter_config.json").exists():
            from peft import PeftModel
            print(f"Usando el adaptador de latín afinado con OPUS ({config.RUTA_ADAPTADOR_LATIN.name}/).")
            self.modelo = PeftModel.from_pretrained(self.modelo, config.RUTA_ADAPTADOR_LATIN)
            self.con_adaptador = True
        self.modelo.eval()

    def _contexto(self, idioma_origen: str, idioma_destino: str, usar_adaptador: bool):
        """El adaptador solo se aprendió con español<->latín: en los demás
        pares se apaga para que el modelo traduzca exactamente como el original."""
        if self.con_adaptador and not (usar_adaptador and "la" in (idioma_origen, idioma_destino)):
            return self.modelo.disable_adapter()
        return nullcontext()

    def _generar(self, textos, idioma_origen, idioma_destino, num_candidatos, usar_adaptador):
        entrada = self.tokenizer(
            [instruccion(self.tokenizer, t, idioma_origen, idioma_destino) for t in textos],
            return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(config.DISPOSITIVO)
        with torch.inference_mode(), self._contexto(idioma_origen, idioma_destino, usar_adaptador):
            salida = self.modelo.generate(
                **entrada,
                max_new_tokens=config.MAX_TOKENS_NUEVOS,
                num_beams=num_candidatos,
                num_return_sequences=num_candidatos,
                do_sample=False,
            )
        # El modelo regresa instrucción + respuesta; solo interesa la respuesta.
        largo_entrada = entrada["input_ids"].shape[1]
        return [t.strip() for t in self.tokenizer.batch_decode(
            salida[:, largo_entrada:], skip_special_tokens=True,
        )]

    def traducir(self, texto: str, idioma_origen: str, idioma_destino: str,
                 num_candidatos: int = 1, usar_adaptador: bool = True) -> list[str]:
        """
        Regresa hasta `num_candidatos` traducciones distintas de `texto`, de la
        más probable a la menos probable según el modelo (beam search).
        """
        candidatos = self._generar([texto], idioma_origen, idioma_destino, num_candidatos, usar_adaptador)
        # El beam search a veces da candidatos repetidos; se quitan sin perder el orden.
        return list(dict.fromkeys(candidatos))

    def traducir_lote(self, textos: list[str], idioma_origen: str, idioma_destino: str,
                      usar_adaptador: bool = True) -> list[str]:
        """Una traducción (la más probable) por texto, procesando varios a la vez."""
        resultado = []
        for i in range(0, len(textos), config.TAM_LOTE_TRADUCCION):
            lote = textos[i:i + config.TAM_LOTE_TRADUCCION]
            resultado += self._generar(lote, idioma_origen, idioma_destino, 1, usar_adaptador)
        return resultado


class Verificador:
    def __init__(self):
        print(f"Cargando verificador de significado '{config.MODELO_SIMILITUD}'...")
        self.modelo = SentenceTransformer(config.MODELO_SIMILITUD, device=str(config.DISPOSITIVO))

    def similitudes(self, referencia: str, frases: list[str]) -> list[float]:
        """Similitud coseno (de -1 a 1) entre `referencia` y cada frase."""
        vectores = self.modelo.encode([referencia, *frases], normalize_embeddings=True)
        return [float(v) for v in vectores[1:] @ vectores[0]]

    def similitudes_por_par(self, frases_a: list[str], frases_b: list[str]) -> list[float]:
        """Similitud coseno entre frases_a[i] y frases_b[i], para cada i."""
        a = self.modelo.encode(frases_a, normalize_embeddings=True, batch_size=64)
        b = self.modelo.encode(frases_b, normalize_embeddings=True, batch_size=64)
        return [float(s) for s in (a * b).sum(axis=1)]
