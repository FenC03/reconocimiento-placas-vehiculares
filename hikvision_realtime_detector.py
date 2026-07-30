# -*- coding: utf-8 -*-
"""
hikvision_realtime_detector.py
--------------------------------
Sistema en tiempo real (un solo proceso, tres hilos) que:

  1. Se conecta al stream RTSP de la camara HikVision (Hilo de Captura).
  2. Cada N segundos, toma un frame y lo encola para analisis.
  3. Un segundo hilo (Hilo de Deteccion) toma esos frames, corre el modelo
     YOLOv8 (.pt) entrenado, y SOLO SI hay detecciones:
       - Guarda el frame completo en DATASET/imagenes/
       - Agrega una fila por cada deteccion en DATASET/detecciones.csv
         (timestamp, archivo de imagen, clase, confianza, bbox x1,y1,x2,y2)
       - Recorta el ROI de cada bbox y lo encola para OCR.
  4. Un tercer hilo (Hilo de OCR) toma esos recortes, aplica el pipeline de
     preprocesamiento (Meesad & Thumthong 2025) y corre Tesseract. Si el
     texto extraido matchea el formato de placa peruana, se imprime en
     pantalla:
         {nombre_archivo.jpg} -> texto detectado : {placa_extraida}
     Si no se detecta ninguna placa valida, no se imprime nada.

Arquitectura: productor/consumidor con colas en memoria (queue.Queue).
La captura del stream NUNCA se bloquea esperando al modelo de deteccion,
ni la deteccion se bloquea esperando al OCR: si un hilo tarda, los items
simplemente se acumulan en su cola y se procesan en cuanto el hilo
consumidor este libre. Si una cola se llena, se descarta el item mas
antiguo para priorizar lo mas reciente.

Uso:
    python hikvision_realtime_detector.py

Se detiene con Ctrl+C (limpia hilos y libera la camara correctamente).
"""

import os
import re
import sys
import csv
import time
import queue
import threading
from datetime import datetime

# -----------------------------------------------------------------------
# IMPORTANTE: configurar ANTES de importar cv2 / crear el VideoCapture.
# Fuerza RTSP sobre TCP (mas estable en Windows / LAN corporativas).
# -----------------------------------------------------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2  # noqa: E402

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] No se encontro el paquete 'ultralytics'. Instalalo con:")
    print("        pip install ultralytics")
    sys.exit(1)

try:
    import pytesseract
except ImportError:
    print("[ERROR] No se encontro el paquete 'pytesseract'. Instalalo con:")
    print("        pip install pytesseract")
    sys.exit(1)


# ============================ CONFIGURACION =============================

# URL RTSP de la camara HikVision.
RTSP_URL = "rtsp://admin:Hik12345@172.16.9.115:554/Streaming/Channels/101"

# Ruta al modelo YOLOv8 entrenado (.pt)
MODEL_PATH = "./modelo-deteccion-placasv3-best.pt"

# Cada cuantos segundos se toma 1 frame para analizar
INTERVALO_SEGUNDOS = 0.01

# Umbral minimo de confianza para considerar una deteccion valida (0.0 - 1.0)
CONFIANZA_MINIMA = 0.5

# Carpetas / archivos de salida (relativos a la ubicacion de este script)
CARPETA_DATASET = "DATASET"
SUBCARPETA_IMAGENES = "imagenes"
ARCHIVO_CSV = "detecciones.csv"

# Calidad de compresion JPG (0-100)
CALIDAD_JPG = 95

# Tamano maximo de la cola en memoria de frames pendientes de detectar.
# Si el detector es mas lento que el intervalo de captura, esto evita que
# la cola crezca sin limite y consuma RAM indefinidamente. Frames mas
# antiguos que no entren se descartan (se prioriza lo mas reciente).
TAMANO_MAXIMO_COLA = 30

# Tamano maximo de la cola en memoria de recortes (ROI) pendientes de OCR.
# Misma logica de descarte que TAMANO_MAXIMO_COLA.
TAMANO_MAXIMO_COLA_OCR = 30

# -------------------------- CONFIGURACION DE OCR (Tesseract) -------------

# Si Tesseract no esta en el PATH del sistema, descomenta y ajusta esta
# ruta (tipico en Windows tras instalar UB-Mannheim/tesseract):
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Regex de validacion para placas peruanas: ej. AYT-573
PERU_PLATE_REGEX = re.compile(r"[A-Z0-9]{3}-?\d{3}")

# Whitelist de caracteres permitidos para Tesseract (mayusculas, digitos y guion)
TESSERACT_CONFIG = (
    "--psm 11 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
)


# ============================ UTILIDADES DE RUTAS =============================

def obtener_ruta_base() -> str:
    """Carpeta donde esta ubicado este script."""
    return os.path.dirname(os.path.abspath(__file__))


def resolver_ruta(ruta: str) -> str:
    """Resuelve una ruta relativa respecto a la ubicacion del script."""
    if os.path.isabs(ruta):
        return ruta
    return os.path.join(obtener_ruta_base(), ruta)


def preparar_carpetas_salida():
    """
    Crea la estructura DATASET/imagenes/ si no existe y devuelve
    (ruta_dataset, ruta_imagenes, ruta_csv).
    """
    ruta_dataset = resolver_ruta(CARPETA_DATASET)
    ruta_imagenes = os.path.join(ruta_dataset, SUBCARPETA_IMAGENES)
    os.makedirs(ruta_imagenes, exist_ok=True)

    ruta_csv = os.path.join(ruta_dataset, ARCHIVO_CSV)

    # Si el CSV no existe aun, lo creamos con encabezados
    if not os.path.isfile(ruta_csv):
        with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "archivo_imagen", "clase", "confianza",
                "x1", "y1", "x2", "y2"
            ])

    return ruta_dataset, ruta_imagenes, ruta_csv


# ============================ DETECCION DE GPU/CPU =============================

def detectar_dispositivo() -> str:
    """
    Detecta automaticamente si hay GPU (CUDA) disponible para usar con YOLO.
    Si no hay, usa CPU sin necesidad de configuracion manual.
    """
    try:
        import torch
        if torch.cuda.is_available():
            nombre_gpu = torch.cuda.get_device_name(0)
            print(f"[INFO] GPU detectada: {nombre_gpu}. Se usara CUDA.")
            return "cuda:0"
    except Exception:
        pass

    print("[INFO] No se detecto GPU disponible. Se usara CPU.")
    return "cpu"


# ============================ UTILIDADES DE OCR =============================

def recortar_roi(imagen, bbox):
    """Recorta la region de interes (placa) de la imagen original."""
    x1, y1, x2, y2 = bbox
    h, w = imagen.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return imagen[y1:y2, x1:x2]


def preprocesar_meesad_thumthong(roi):
    """
    Pipeline de Meesad & Thumthong (2025): escala de grises ->
    denoising (mediana + gaussiano) -> ecualizacion de histograma ->
    binarizacion Otsu.
    """
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    denoised = cv2.medianBlur(gris, 3)
    denoised = cv2.GaussianBlur(denoised, (3, 3), 0)

    contrastada = cv2.equalizeHist(denoised)

    _, binaria = cv2.threshold(
        contrastada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return binaria


def extraer_texto_placa(imagen_procesada):
    """
    Ejecuta Tesseract y valida el resultado contra el formato de placa
    peruana. Devuelve el substring que matchea, o None si no hay match.
    """
    texto_crudo = pytesseract.image_to_string(
        imagen_procesada, config=TESSERACT_CONFIG
    )

    texto_crudo = texto_crudo.strip().replace(" ", "").replace("\n", "")

    match = PERU_PLATE_REGEX.search(texto_crudo)
    return match.group(0) if match else None


# ============================ HILO 1: CAPTURA =============================

def hilo_captura(cola_frames: queue.Queue, evento_detener: threading.Event):
    """
    Se conecta al stream RTSP y empuja 1 frame por intervalo a la cola.
    Reconecta automaticamente si se pierde la senal. Nunca se bloquea
    esperando al hilo de deteccion: si la cola esta llena, descarta el
    frame mas antiguo para dejar espacio al mas reciente.
    """
    cap = None

    while not evento_detener.is_set():
        if cap is None:
            print("[CAPTURA] Conectando a la camara...")
            cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            if not cap.isOpened():
                print("[CAPTURA][ERROR] No se pudo conectar. Reintentando en 3s...")
                cap.release()
                cap = None
                time.sleep(3)
                continue

            print("[CAPTURA] Conexion establecida.")

        ultimo_envio = 0.0
        fallos_seguidos = 0

        while not evento_detener.is_set():
            ret, frame = cap.read()

            if not ret or frame is None:
                fallos_seguidos += 1
                if fallos_seguidos >= 10:
                    print("[CAPTURA][AVISO] Senal perdida. Reconectando...")
                    cap.release()
                    cap = None
                    break
                continue

            fallos_seguidos = 0
            ahora = time.time()

            if ahora - ultimo_envio >= INTERVALO_SEGUNDOS:
                ultimo_envio = ahora
                timestamp = datetime.now()

                # Si la cola esta llena, descartamos el frame mas viejo
                # para no acumular retraso indefinidamente.
                if cola_frames.full():
                    try:
                        cola_frames.get_nowait()
                    except queue.Empty:
                        pass

                cola_frames.put((timestamp, frame.copy()))

        if cap is None:
            # Pequena pausa antes de reintentar conexion, para no saturar
            # la red/CPU con reconexiones inmediatas si la camara esta
            # realmente caida (no solo un corte momentaneo).
            time.sleep(2)

    if cap is not None:
        cap.release()
    print("[CAPTURA] Hilo de captura finalizado.")


# ============================ HILO 2: DETECCION =============================

def hilo_deteccion(cola_frames: queue.Queue, cola_ocr: queue.Queue,
                    evento_detener: threading.Event,
                    modelo, dispositivo: str, ruta_imagenes: str, ruta_csv: str):
    """
    Toma frames de la cola, corre el modelo YOLOv8, y si hay detecciones
    con confianza suficiente, guarda el frame + registra metadatos en CSV.
    Ademas, recorta el ROI de cada deteccion y lo encola para OCR.
    """
    contador_detecciones = 0

    while not evento_detener.is_set():
        try:
            timestamp, frame = cola_frames.get(timeout=1.0)
        except queue.Empty:
            continue

        resultados = modelo.predict(
            frame,
            device=dispositivo,
            conf=CONFIANZA_MINIMA,
            verbose=False
        )

        resultado = resultados[0]
        cajas = resultado.boxes

        if cajas is None or len(cajas) == 0:
            # Sin detecciones en este frame: no se guarda nada (dataset solo positivos)
            continue

        # --- Hay al menos una deteccion: guardamos el frame ---
        marca = timestamp.strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]  # incluye milisegundos
        nombre_imagen = f"deteccion_{marca}.jpg"
        ruta_imagen_completa = os.path.join(ruta_imagenes, nombre_imagen)

        cv2.imwrite(ruta_imagen_completa, frame, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPG])

        # --- Registramos cada deteccion individual en el CSV y encolamos su ROI para OCR ---
        nombres_clases = modelo.names
        filas_nuevas = []

        for caja in cajas:
            x1, y1, x2, y2 = caja.xyxy[0].tolist()
            confianza = float(caja.conf[0])
            id_clase = int(caja.cls[0])
            nombre_clase = nombres_clases.get(id_clase, str(id_clase))

            print(f"[DETECCION] Placa detectada -> clase: {nombre_clase}, confianza: {confianza:.2f}")

            filas_nuevas.append([
                timestamp.strftime("%Y-%m-%d %H:%M:%S.%f"),
                nombre_imagen,
                nombre_clase,
                f"{confianza:.4f}",
                f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"
            ])

            # Recortamos el ROI de esta deteccion y lo enviamos al hilo de OCR.
            roi = recortar_roi(frame, (int(x1), int(y1), int(x2), int(y2)))
            if roi.size > 0:
                if cola_ocr.full():
                    try:
                        cola_ocr.get_nowait()
                    except queue.Empty:
                        pass
                cola_ocr.put((nombre_imagen, roi))

        with open(ruta_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(filas_nuevas)

        contador_detecciones += 1
        print(f"[DETECCION] #{contador_detecciones} -> {len(filas_nuevas)} objeto(s) "
              f"en {nombre_imagen}")

    print("[DETECCION] Hilo de deteccion finalizado.")


# ============================ HILO 3: OCR =============================

def hilo_ocr(cola_ocr: queue.Queue, evento_detener: threading.Event):
    """
    Toma recortes (ROI) de la cola, aplica el preprocesamiento de
    Meesad & Thumthong y corre Tesseract. Si el texto extraido matchea
    el formato de placa peruana, lo imprime en pantalla. Si no hay
    match, no imprime nada.
    """
    while not evento_detener.is_set():
        try:
            nombre_imagen, roi = cola_ocr.get(timeout=1.0)
        except queue.Empty:
            continue

        try:
            imagen_procesada = preprocesar_meesad_thumthong(roi)
            placa_extraida = extraer_texto_placa(imagen_procesada)
        except Exception as e:
            print(f"[OCR][ERROR] Fallo procesando {nombre_imagen}: {e}")
            continue

        if placa_extraida:
            print(f"{nombre_imagen} -> texto detectado : {placa_extraida}")

    print("[OCR] Hilo de OCR finalizado.")


# ============================ MAIN =============================

def main():
    print("=" * 60)
    print(" Sistema en tiempo real: Captura + Deteccion YOLOv8 + OCR")
    print("=" * 60)
    print(f" Camara:    {RTSP_URL}")
    print(f" Modelo:    {MODEL_PATH}")
    print(f" Intervalo: 1 frame cada {INTERVALO_SEGUNDOS} segundo(s)")
    print(f" Confianza minima: {CONFIANZA_MINIMA}")
    print(" Presiona Ctrl+C para detener")
    print("=" * 60)

    ruta_modelo = resolver_ruta(MODEL_PATH)
    if not os.path.isfile(ruta_modelo):
        print(f"[ERROR] No se encontro el modelo en: {ruta_modelo}")
        print("        Verifica la variable MODEL_PATH en el script.")
        sys.exit(1)

    print("[INFO] Cargando modelo YOLOv8...")
    dispositivo = detectar_dispositivo()
    modelo = YOLO(ruta_modelo)

    ruta_dataset, ruta_imagenes, ruta_csv = preparar_carpetas_salida()
    print(f"[INFO] Dataset se guardara en: {ruta_dataset}")

    cola_frames = queue.Queue(maxsize=TAMANO_MAXIMO_COLA)
    cola_ocr = queue.Queue(maxsize=TAMANO_MAXIMO_COLA_OCR)
    evento_detener = threading.Event()

    t_captura = threading.Thread(
        target=hilo_captura, args=(cola_frames, evento_detener), daemon=True
    )
    t_deteccion = threading.Thread(
        target=hilo_deteccion,
        args=(cola_frames, cola_ocr, evento_detener, modelo, dispositivo,
              ruta_imagenes, ruta_csv),
        daemon=True
    )
    t_ocr = threading.Thread(
        target=hilo_ocr, args=(cola_ocr, evento_detener), daemon=True
    )

    t_captura.start()
    t_deteccion.start()
    t_ocr.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Deteniendo sistema (Ctrl+C detectado)...")
    finally:
        evento_detener.set()
        t_captura.join(timeout=5)
        t_deteccion.join(timeout=5)
        t_ocr.join(timeout=5)
        print("[INFO] Sistema detenido correctamente.")
        print(f"[INFO] Revisa el dataset generado en: {ruta_dataset}")


if __name__ == "__main__":
    main()
