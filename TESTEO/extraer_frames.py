import cv2
import os

# ============================================================
#  CONFIGURACIÓN
# ============================================================

# Ruta del video a procesar
VIDEO_PATH = "./VIDEOS/grabacion_2026-06-19_12-59-36.mp4"

# Carpeta donde se guardarán las imágenes extraídas
OUTPUT_DIR = "IMAGENES/"

# FPS de respaldo: se usa SOLO si OpenCV no puede leer los FPS del video.
# Ajusta este valor si sabes la tasa de tu cámara (ej. 24, 30, 60).
FPS_FALLBACK = 30

# Formato de las imágenes guardadas ("jpg" o "png")
IMAGE_FORMAT = "jpg"

# ============================================================
#  EXTRACCIÓN DE FRAMES
# ============================================================

def extraer_primer_frame_por_segundo(video_path, output_dir, fps_fallback, image_format):
    # Verificar que el archivo de video existe
    if not os.path.isfile(video_path):
        print(f"[ERROR] No se encontró el video: '{video_path}'")
        return

    # Crear carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el video: '{video_path}'")
        return

    # Obtener FPS del video automáticamente
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[AVISO] No se pudieron detectar los FPS del video. Usando valor de respaldo: {fps_fallback} FPS.")
        fps = fps_fallback
    else:
        print(f"[INFO] FPS detectados automáticamente: {fps:.2f}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_seg = int(total_frames / fps)
    print(f"[INFO] Total de frames: {total_frames} | Duración aproximada: {duracion_seg} segundos")
    print(f"[INFO] Guardando imágenes en: '{output_dir}'")
    print("-" * 50)

    frames_guardados = 0
    segundo_actual = 0

    while True:
        # Posicionarse en el frame exacto del segundo actual
        frame_objetivo = int(segundo_actual * fps)

        if frame_objetivo >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_objetivo)
        ret, frame = cap.read()

        if not ret:
            break

        # Guardar la imagen
        nombre_archivo = f"frame_seg_{segundo_actual:05d}.{image_format}"
        ruta_salida = os.path.join(output_dir, nombre_archivo)
        cv2.imwrite(ruta_salida, frame)

        print(f"  Segundo {segundo_actual:>5} → {nombre_archivo}")
        frames_guardados += 1
        segundo_actual += 1

    cap.release()
    print("-" * 50)
    print(f"[OK] Proceso terminado. Se guardaron {frames_guardados} imágenes en '{output_dir}'.")


if __name__ == "__main__":
    extraer_primer_frame_por_segundo(VIDEO_PATH, OUTPUT_DIR, FPS_FALLBACK, IMAGE_FORMAT)
