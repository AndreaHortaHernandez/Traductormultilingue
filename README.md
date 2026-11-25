# Práctica 2 · Reconocimiento de rostros: CNN desde cero vs MobileNetV2

Rama aparte del repo, solo con el notebook de entrenamiento (sin fotos: los datos se quedan en el Drive del equipo).

**Abrir en Colab:**
https://colab.research.google.com/github/AndreaHortaHernandez/Traductormultilingue/blob/p2-rostros/colab/CNN_vs_MobileNet_v2_Colab.ipynb

Reentrena la CNN desde cero con la misma arquitectura que `models/keras_scratch.keras`, corrigiendo el entrenamiento
(coseno en vez de `ReduceLROnPlateau`, 120 épocas, BatchNorm `momentum=0.9`), y MobileNetV2 con la misma división.
Lee `escom_caras/caras_manifest.csv` y `escom_caras/caras/` del Drive (agrega el acceso directo a *Mi unidad*).
