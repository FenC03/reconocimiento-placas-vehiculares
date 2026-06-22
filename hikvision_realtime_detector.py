# -*- coding: utf-8 -*-
"""
hikvision_realtime_detector.py
--------------------------------
Sistema en tiempo real (un solo proceso, dos hilos) que:

  1. Se conecta al stream RTSP de la camara HikVision (Hilo de Captura).
  2. Cada 1 segundo, toma un frame y lo encola para analisis.
  3. Un segundo hilo (Hilo de Deteccion) toma esos frames, corre el modelo
     YOLOv8 (.pt) entrenado, y SOLO SI hay detecciones:
       - Guarda el frame completo en DATASET/imagenes/
       - Agrega una fila por cada deteccion en DATASET/detecciones.csv
         (timestamp, archivo de imagen, clase, confianza, bbox x1,y1,x2,y2)

Arquitectura: productor/consumidor con cola en memoria (queue.Queue).
La captura del stream NUNCA se bloquea esperando al modelo: si el modelo
tarda, los frames simplemente se acumulan en la cola y se procesan en
cuanto el detector este libre.

Uso:
    python hikvision_realtime_detector.py

Se detiene con Ctrl+C (limpia hilos y libera la camara correctamente).
"""

import os
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


# ============================ CONFIGURACION =============================

# URL RTSP de la camara HikVision.
RTSP_URL = "rtsp://admin:Hik12345@172.16.9.115:554/Streaming/Channels/101"

# Ruta al modelo YOLOv8 entrenado (.pt)
MODEL_PATH = "./best.pt"

# Cada cuantos segundos se toma 1 frame para analizar
INTERVALO_SEGUNDOS = 1.0

# Umbral minimo de confianza para considerar una deteccion valida (0.0 - 1.0)
CONFIANZA_MINIMA = 0.5

# Carpetas / archivos de salida (relativos a la ubicacion de este script)
CARPETA_DATASET = "DATASET"
SUBCARPETA_IMAGENES = "imagenes"
ARCHIVO_CSV = "detecciones.csv"

# Calidad de compresion JPG (0-100)
CALIDAD_JPG = 95

# Tamano maximo de la cola en memoria (frames pendientes de analizar).
# Si el detector es mas lento que 1 frame/seg, esto evita que la cola
# crezca sin limite y consuma RAM indefinidamente. Frames mas antiguos
# que no entren se descartan (se prioriza lo mas reciente).
TAMANO_MAXIMO_COLA = 30


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


# ============================ HILO 1: CAPTURA =============================

def hilo_captura(cola_frames: queue.Queue, evento_detener: threading.Event):
    """
    Se conecta al stream RTSP y empuja 1 frame por segundo a la cola.
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

def hilo_deteccion(cola_frames: queue.Queue, evento_detener: threading.Event,
                    modelo, dispositivo: str, ruta_imagenes: str, ruta_csv: str):
    """
    Toma frames de la cola, corre el modelo YOLOv8, y si hay detecciones
    con confianza suficiente, guarda el frame + registra metadatos en CSV.
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

        # --- Registramos cada deteccion individual en el CSV ---
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

        with open(ruta_csv, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(filas_nuevas)

        contador_detecciones += 1
        print(f"[DETECCION] #{contador_detecciones} -> {len(filas_nuevas)} objeto(s) "
              f"en {nombre_imagen}")

    print("[DETECCION] Hilo de deteccion finalizado.")


# ============================ MAIN =============================

def main():
    print("=" * 60)
    print(" Sistema en tiempo real: Captura + Deteccion YOLOv8")
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
    evento_detener = threading.Event()

    t_captura = threading.Thread(
        target=hilo_captura, args=(cola_frames, evento_detener), daemon=True
    )
    t_deteccion = threading.Thread(
        target=hilo_deteccion,
        args=(cola_frames, evento_detener, modelo, dispositivo, ruta_imagenes, ruta_csv),
        daemon=True
    )

    t_captura.start()
    t_deteccion.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Deteniendo sistema (Ctrl+C detectado)...")
    finally:
        evento_detener.set()
        t_captura.join(timeout=5)
        t_deteccion.join(timeout=5)
        print("[INFO] Sistema detenido correctamente.")
        print(f"[INFO] Revisa el dataset generado en: {ruta_dataset}")


if __name__ == "__main__":
    main()
