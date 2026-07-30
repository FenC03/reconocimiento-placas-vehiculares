import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image as PILImage
import numpy as np
import cv2

# ============================================================
#  CONFIGURACIÓN
# ============================================================

# Ruta del modelo YOLOv8 entrenado
MODEL_PATH = "best.pt"

# Carpeta con las imágenes a analizar
INPUT_DIR = "IMAGENES/"

# Carpeta donde se guardan las imágenes donde se detectó placa
OUTPUT_DIR = "PLACAS DETECTADAS/"

# Confianza mínima para considerar una detección válida (0.0 - 1.0)
CONF_THRESH = 0.45

# Extensiones de imagen a procesar
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp", ".tiff"}

# ============================================================
#  FUNCIÓN: cargar imagen compatible con múltiples formatos
# ============================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Carga imagen con PIL y la convierte a BGR para OpenCV/YOLO.
    PIL acepta .jfif y otros formatos que cv2.imread rechaza.
    """
    pil_img = PILImage.open(image_path).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ============================================================
#  PIPELINE PRINCIPAL
# ============================================================

def detectar_placas(model_path, input_dir, output_dir, conf_thresh):
    # Verificar que la carpeta de entrada exista
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[ERROR] No se encontró la carpeta de entrada: '{input_dir}'")
        return

    # Crear carpeta de salida si no existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Cargar modelo YOLOv8
    print(f"[INFO] Cargando modelo: {model_path}")
    if not Path(model_path).is_file():
        print(f"[ERROR] No se encontró el modelo: '{model_path}'")
        return
    model = YOLO(model_path)
    print(f"[INFO] Modelo cargado correctamente.\n")

    # Obtener lista de imágenes válidas
    imagenes = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
    ]

    if not imagenes:
        print(f"[AVISO] No se encontraron imágenes en '{input_dir}'.")
        return

    print(f"[INFO] {len(imagenes)} imágenes encontradas en '{input_dir}'.")
    print(f"[INFO] Umbral de confianza: {conf_thresh}")
    print("-" * 55)

    total_con_placa    = 0
    total_sin_placa    = 0
    total_con_error    = 0

    for imagen_path in sorted(imagenes):
        nombre = imagen_path.name

        try:
            frame = load_image(str(imagen_path))
        except Exception as e:
            print(f"  [ERROR] {nombre} → no se pudo leer: {e}")
            total_con_error += 1
            continue

        # Inferencia YOLO
        resultado = model(frame, conf=conf_thresh, verbose=False)[0]
        n_det = len(resultado.boxes)

        if n_det > 0:
            # Hay al menos una placa → copiar imagen original a la carpeta de salida
            destino = output_path / nombre
            shutil.copy2(str(imagen_path), str(destino))
            print(f"  ✔ PLACA DETECTADA  ({n_det} detección/es) → {nombre}")
            total_con_placa += 1
        else:
            print(f"  ✘ Sin placa                              → {nombre}")
            total_sin_placa += 1

    # Resumen final
    print("-" * 55)
    print(f"[RESUMEN]")
    print(f"  Imágenes procesadas  : {len(imagenes)}")
    print(f"  Con placa detectada  : {total_con_placa}  → guardadas en '{output_dir}'")
    print(f"  Sin placa            : {total_sin_placa}")
    if total_con_error:
        print(f"  Con error de lectura : {total_con_error}")
    print(f"\n[OK] Proceso terminado.")


# ============================================================
#  ENTRADA
# ============================================================

if __name__ == "__main__":
    detectar_placas(MODEL_PATH, INPUT_DIR, OUTPUT_DIR, CONF_THRESH)
