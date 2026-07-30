# -*- coding: utf-8 -*-
"""
hikvision_viewer.py
--------------------
Aplicacion de escritorio simple (Windows) para visualizar en tiempo real
el stream RTSP de una camara HikVision usando OpenCV.

Uso:
    python hikvision_viewer.py

Controles:
    Q       -> Cierra la aplicacion
    (cerrar la ventana tambien funciona)

Autor: Generado con Claude
"""

import os
import sys
import time

# -----------------------------------------------------------------------
# IMPORTANTE: esta variable de entorno debe configurarse ANTES de importar
# cv2 / crear el VideoCapture. Fuerza a FFMPEG a usar RTSP sobre TCP en vez
# de UDP, lo cual es mucho mas estable en redes Windows / LAN corporativas
# y evita el clasico problema de imagen verde / cuadros corruptos.
# -----------------------------------------------------------------------
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2  # noqa: E402  (import despues de configurar la variable de entorno)


# ============================ CONFIGURACION =============================

# URL RTSP de la camara HikVision.
# Formato HikVision: rtsp://usuario:password@IP:puerto/Streaming/Channels/<canal>
#   - Canal 101 = Camara 1, stream principal (main stream, mayor calidad)
#   - Canal 102 = Camara 1, sub stream (menor calidad, mas liviano)
RTSP_URL = "rtsp://admin:Hik12345@172.16.9.115:554/Streaming/Channels/101"

# Nombre de la ventana que se mostrara en Windows
WINDOW_NAME = "HikVision - Stream en vivo (Q para salir)"

# Tiempo de espera (segundos) antes de reintentar si se pierde la conexion
RECONNECT_DELAY = 3

# Numero maximo de intentos de lectura fallidos antes de considerar
# que se perdio la conexion y se debe reconectar
MAX_FAILED_READS = 10

# Ancho maximo de la ventana (si el video es mas grande, se reescala
# para que no ocupe toda la pantalla). Usa None para mostrar a tamano
# original.
MAX_DISPLAY_WIDTH = 1280


# ============================ FUNCIONES =============================

def conectar_stream(url: str):
    """
    Intenta abrir el stream RTSP. Devuelve el objeto VideoCapture
    si tuvo exito, o None si fallo.
    """
    print(f"[INFO] Conectando a la camara...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    # Reduce el buffer interno para minimizar el delay (latencia) del video.
    # No todos los backends respetan esto, pero no hace dano intentarlo.
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

    print("[OK] Conexion establecida. Mostrando video...")
    return cap


def redimensionar_si_necesario(frame):
    """
    Si el frame es mas ancho que MAX_DISPLAY_WIDTH, lo reescala
    manteniendo la proporcion (aspect ratio).
    """
    if MAX_DISPLAY_WIDTH is None:
        return frame

    alto, ancho = frame.shape[:2]
    if ancho <= MAX_DISPLAY_WIDTH:
        return frame

    escala = MAX_DISPLAY_WIDTH / float(ancho)
    nuevo_alto = int(alto * escala)
    return cv2.resize(frame, (MAX_DISPLAY_WIDTH, nuevo_alto), interpolation=cv2.INTER_AREA)


def main():
    print("=" * 60)
    print(" Visor de stream HikVision - RTSP")
    print("=" * 60)
    print(f" URL: {RTSP_URL}")
    print(" Presiona 'Q' en la ventana de video para salir")
    print("=" * 60)

    cap = None

    try:
        while True:
            # --- Conexion / reconexion ---
            if cap is None:
                cap = conectar_stream(RTSP_URL)
                if cap is None:
                    print(f"[INFO] Reintentando en {RECONNECT_DELAY} segundos...")
                    time.sleep(RECONNECT_DELAY)
                    continue

            fallos_seguidos = 0

            # --- Loop de lectura de frames ---
            while True:
                ret, frame = cap.read()

                if not ret or frame is None:
                    fallos_seguidos += 1
                    if fallos_seguidos >= MAX_FAILED_READS:
                        print("[AVISO] Se perdio la senal de la camara. Reconectando...")
                        cap.release()
                        cap = None
                        break
                    continue

                fallos_seguidos = 0
                frame_mostrar = redimensionar_si_necesario(frame)

                cv2.imshow(WINDOW_NAME, frame_mostrar)

                # waitKey(1) -> revisa cada ~1ms si se presiono una tecla
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord('q') or tecla == ord('Q'):
                    print("[INFO] Cierre solicitado por el usuario.")
                    raise KeyboardInterrupt

                # Si el usuario cerro la ventana con la X
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Ventana cerrada por el usuario.")
                    raise KeyboardInterrupt

            if cap is None:
                time.sleep(RECONNECT_DELAY)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] Ocurrio un error inesperado: {e}")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Aplicacion cerrada correctamente.")


if __name__ == "__main__":
    main()
