# -*- coding: utf-8 -*-
"""
hikvision_recorder.py
----------------------
Aplicacion de escritorio simple (Windows) que se conecta al stream RTSP
de una camara HikVision, graba 30 segundos de video y lo guarda como
archivo .mp4 dentro de la carpeta "VIDEOS/".

Uso:
    python hikvision_recorder.py

El script termina automaticamente al completar los 30 segundos
(o se puede cancelar antes presionando 'Q' en la ventana de previsualizacion).
"""

import os
import sys
import time
from datetime import datetime

# -----------------------------------------------------------------------
# IMPORTANTE: configurar ANTES de importar cv2 / crear el VideoCapture.
# Fuerza RTSP sobre TCP (mas estable en Windows / LAN corporativas).
# -----------------------------------------------------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2  # noqa: E402


# ============================ CONFIGURACION =============================

# URL RTSP de la camara HikVision.
RTSP_URL = "rtsp://admin:Hik12345@172.16.9.115:554/Streaming/Channels/101"

# Duracion de la grabacion en segundos
DURACION_SEGUNDOS = 60

# Carpeta donde se guardaran los videos (relativa a la ubicacion de este script)
CARPETA_VIDEOS = "VIDEOS"

# FPS de respaldo si la camara no reporta un valor valido via RTSP
FPS_FALLBACK = 20

# Si se muestra una ventana de previsualizacion mientras se graba.
# Util para confirmar visualmente que se esta capturando lo correcto.
MOSTRAR_PREVIEW = True
WINDOW_NAME = "Grabando... (Q para cancelar)"

# Codec de video. 'mp4v' es el mas compatible con OpenCV en Windows sin
# necesitar codecs externos adicionales.
FOURCC = "mp4v"
EXTENSION = ".mp4"


# ============================ FUNCIONES =============================

def obtener_ruta_base() -> str:
    """
    Devuelve la carpeta donde esta ubicado este script (no el directorio
    desde donde se ejecuta), para que VIDEOS/ siempre se cree en el lugar
    correcto sin importar como se invoque el script.
    """
    return os.path.dirname(os.path.abspath(__file__))


def asegurar_carpeta_videos() -> str:
    """
    Crea (si no existe) la carpeta VIDEOS/ junto al script y devuelve su ruta.
    """
    ruta_videos = os.path.join(obtener_ruta_base(), CARPETA_VIDEOS)
    os.makedirs(ruta_videos, exist_ok=True)
    return ruta_videos


def generar_nombre_archivo() -> str:
    """
    Genera un nombre de archivo unico basado en fecha y hora actual,
    para no sobrescribir grabaciones anteriores.
    """
    marca_tiempo = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"grabacion_{marca_tiempo}{EXTENSION}"


def conectar_stream(url: str):
    """
    Intenta abrir el stream RTSP. Devuelve el objeto VideoCapture
    si tuvo exito, o None si fallo.
    """
    print("[INFO] Conectando a la camara...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir el stream. Verifica:")
        print("        - Que la camara este encendida y en la misma red")
        print("        - Que la IP, usuario y password sean correctos")
        print("        - Que el puerto 554 no este bloqueado por un firewall")
        cap.release()
        return None

    print("[OK] Conexion establecida.")
    return cap


def obtener_fps_valido(cap) -> float:
    """
    Obtiene el FPS reportado por el stream. Si no es un valor valido
    (0, negativo, o NaN -- algo comun en RTSP), usa el valor de respaldo.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps != fps:  # fps != fps detecta NaN
        print(f"[AVISO] La camara no reporto un FPS valido. Usando {FPS_FALLBACK} FPS por defecto.")
        return float(FPS_FALLBACK)
    return float(fps)


def main():
    print("=" * 60)
    print(" Grabador de stream HikVision - RTSP")
    print("=" * 60)
    print(f" URL: {RTSP_URL}")
    print(f" Duracion: {DURACION_SEGUNDOS} segundos")
    print("=" * 60)

    ruta_videos = asegurar_carpeta_videos()
    nombre_archivo = generar_nombre_archivo()
    ruta_completa = os.path.join(ruta_videos, nombre_archivo)

    cap = conectar_stream(RTSP_URL)
    if cap is None:
        print("[ERROR] No se pudo iniciar la grabacion. Saliendo.")
        sys.exit(1)

    # --- Leer un primer frame para obtener dimensiones reales del video ---
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] No se pudo leer ningun frame de la camara. Saliendo.")
        cap.release()
        sys.exit(1)

    alto, ancho = frame.shape[:2]
    fps = obtener_fps_valido(cap)

    print(f"[INFO] Resolucion detectada: {ancho}x{alto} @ {fps:.1f} FPS")
    print(f"[INFO] Guardando en: {ruta_completa}")

    # --- Preparar el escritor de video ---
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(ruta_completa, fourcc, fps, (ancho, alto))

    if not writer.isOpened():
        print("[ERROR] No se pudo crear el archivo de video. Verifica permisos de escritura.")
        cap.release()
        sys.exit(1)

    if MOSTRAR_PREVIEW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("[INFO] Grabando... presiona 'Q' en la ventana de preview para cancelar.")

    tiempo_inicio = time.time()
    frames_escritos = 0
    cancelado = False

    try:
        # Escribimos el primer frame ya leido arriba
        writer.write(frame)
        frames_escritos += 1
        if MOSTRAR_PREVIEW:
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)

        while True:
            tiempo_transcurrido = time.time() - tiempo_inicio
            if tiempo_transcurrido >= DURACION_SEGUNDOS:
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[AVISO] Frame perdido durante la grabacion, se omite.")
                continue

            writer.write(frame)
            frames_escritos += 1

            if MOSTRAR_PREVIEW:
                # Muestra segundos restantes en consola cada vez que cambia el segundo
                segundos_restantes = max(0, int(DURACION_SEGUNDOS - tiempo_transcurrido))
                texto = f"Grabando... {segundos_restantes}s restantes"
                cv2.putText(frame, texto, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, frame)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord('q') or tecla == ord('Q'):
                    print("[INFO] Grabacion cancelada por el usuario.")
                    cancelado = True
                    break

    except KeyboardInterrupt:
        print("[INFO] Grabacion interrumpida por el usuario (Ctrl+C).")
        cancelado = True
    finally:
        cap.release()
        writer.release()
        if MOSTRAR_PREVIEW:
            cv2.destroyAllWindows()

    duracion_real = time.time() - tiempo_inicio

    print("=" * 60)
    if cancelado:
        print(f"[INFO] Grabacion cancelada. Se guardaron {frames_escritos} frames "
              f"({duracion_real:.1f} segundos) en:")
    else:
        print(f"[OK] Grabacion completada. {frames_escritos} frames "
              f"({duracion_real:.1f} segundos) guardados en:")
    print(f"     {ruta_completa}")
    print("=" * 60)


if __name__ == "__main__":
    main()
