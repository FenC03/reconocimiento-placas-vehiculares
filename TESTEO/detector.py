"""
detector.py — Worker permanente de detección ALPR
Carga YOLOv8 + EasyOCR UNA sola vez al arrancar.
Recibe frames via queue.Queue y devuelve resultados via results_queue.
"""

import re
import queue
import threading
import time
import cv2
import numpy as np
from pathlib import Path

# ── Lazy imports: solo se importan si el worker arranca ──────────────────────
_yolo_model  = None
_ocr_reader  = None
_model_lock  = threading.Lock()

MODEL_WEIGHTS = "best.pt"   # Cambiar a la ruta de tu best.pt
CONF_THRESH   = 0.45


# ─────────────────────────────────────────────────────────────────────────────
# Carga de modelos (singleton, thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
def _load_models():
    global _yolo_model, _ocr_reader
    with _model_lock:
        if _yolo_model is None:
            from ultralytics import YOLO
            print("[DETECTOR] Cargando YOLOv8...")
            _yolo_model = YOLO(MODEL_WEIGHTS)
            print("[DETECTOR] YOLOv8 listo.")

        if _ocr_reader is None:
            import easyocr
            print("[DETECTOR] Cargando EasyOCR (puede tardar ~10s la primera vez)...")
            _ocr_reader = easyocr.Reader(["en"], gpu=False)
            print("[DETECTOR] EasyOCR listo.")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de detección (igual a tu notebook, adaptado a función pura)
# ─────────────────────────────────────────────────────────────────────────────
def _crop_plate(frame: np.ndarray, box, pad: int = 4) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(frame.shape[1], x2 + pad)
    y2p = min(frame.shape[0], y2 + pad)
    return frame[y1p:y2p, x1p:x2p]


def _preprocess_plate(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    crop = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=8
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _clean_plate_text(raw: str) -> str:
    text = raw.upper().replace(" ", "").replace("-", "")
    if len(text) < 6:
        return raw.upper()
    result = list(text[:6])
    letter_fixes = {"0": "O", "1": "I", "8": "B"}
    number_fixes = {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}
    for idx in range(3):
        result[idx] = letter_fixes.get(result[idx], result[idx])
    for idx in range(3, 6):
        result[idx] = number_fixes.get(result[idx], result[idx])
    cleaned = "".join(result)
    if re.match(r"^[A-Z]{3}\d{3}$", cleaned):
        return f"{cleaned[:3]}-{cleaned[3:]}"
    return raw.upper()


def _read_plate(crop: np.ndarray) -> dict:
    processed = _preprocess_plate(crop)
    results = _ocr_reader.readtext(
        processed,
        detail=1,
        paragraph=False,
        width_ths=0.7,
        decoder="beamsearch",
        beamWidth=10,
        text_threshold=0.5,
        low_text=0.3,
    )
    if not results:
        return {"plate": "", "conf": 0.0}
    texts = [t for (_, t, _) in results]
    confs = [c for (_, _, c) in results]
    raw_text = " ".join(texts)
    cleaned  = _clean_plate_text(raw_text)
    return {"plate": cleaned, "conf": round(sum(confs) / len(confs), 4)}


def detect_frame(frame: np.ndarray) -> list[dict]:
    """
    Recibe un frame BGR de OpenCV.
    Devuelve lista de detecciones:
      [{"plate": "ABC-123", "conf_ocr": 0.92, "conf_det": 0.87, "bbox": [x1,y1,x2,y2]}, ...]
    """
    yolo_results = _yolo_model(frame, conf=CONF_THRESH, verbose=False)[0]
    detections = []
    for box_data in yolo_results.boxes:
        box      = box_data.xyxy[0].cpu().numpy()
        det_conf = float(box_data.conf[0])
        crop     = _crop_plate(frame, box)
        ocr      = _read_plate(crop)
        detections.append({
            "plate":    ocr["plate"],
            "conf_ocr": ocr["conf"],
            "conf_det": det_conf,
            "bbox":     list(map(int, box)),
        })
    return detections


# ─────────────────────────────────────────────────────────────────────────────
# Worker permanente
# ─────────────────────────────────────────────────────────────────────────────
class DetectorWorker:
    """
    Se instancia una vez. Vive durante toda la ejecución de la app.
    
    Uso:
        worker = DetectorWorker(max_queue=30)
        worker.start()
        worker.submit(frame_id, frame_bgr)
        results = worker.get_results(max_items=10)
    """

    def __init__(self, max_queue: int = 30):
        self._frame_queue   = queue.Queue(maxsize=max_queue)
        self._results       = []          # lista compartida de detecciones
        self._results_lock  = threading.Lock()
        self._thread        = None
        self._running       = False
        self.ready          = False       # True cuando los modelos están cargados
        self.frames_processed = 0
        self.frames_dropped   = 0

    def start(self):
        """Arranca el hilo del worker en background."""
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True, name="DetectorWorker")
        self._thread.start()

    def stop(self):
        self._running = False

    def submit(self, frame_id: str, frame: np.ndarray) -> bool:
        """
        Encola un frame para detección.
        Si la cola está llena, descarta el frame (no bloquea la grabación).
        Devuelve True si fue encolado, False si fue descartado.
        """
        try:
            self._frame_queue.put_nowait((frame_id, frame))
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False

    def get_results(self, max_items: int = 50) -> list:
        """Retorna las últimas detecciones (thread-safe)."""
        with self._results_lock:
            return list(self._results[-max_items:])

    def get_stats(self) -> dict:
        return {
            "ready":            self.ready,
            "queue_size":       self._frame_queue.qsize(),
            "frames_processed": self.frames_processed,
            "frames_dropped":   self.frames_dropped,
        }

    def _run(self):
        """Loop principal del worker — se ejecuta en su propio hilo."""
        print("[DETECTOR] Iniciando carga de modelos...")
        try:
            _load_models()
            self.ready = True
            print("[DETECTOR] Worker listo para recibir frames.")
        except Exception as e:
            print(f"[DETECTOR] ERROR al cargar modelos: {e}")
            self._running = False
            return

        while self._running:
            try:
                frame_id, frame = self._frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                t0 = time.time()
                detections = detect_frame(frame)
                elapsed_ms = round((time.time() - t0) * 1000)

                entry = {
                    "frame_id":   frame_id,
                    "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
                    "elapsed_ms": elapsed_ms,
                    "detections": detections,
                }

                with self._results_lock:
                    self._results.append(entry)
                    # Mantener solo los últimos 500 resultados en memoria
                    if len(self._results) > 500:
                        self._results = self._results[-500:]

                self.frames_processed += 1

                if detections:
                    plates = [d["plate"] for d in detections if d["plate"]]
                    print(f"[DETECTOR] {frame_id} → {plates} ({elapsed_ms}ms)")

            except Exception as e:
                print(f"[DETECTOR] Error procesando {frame_id}: {e}")


# Instancia global — única durante toda la vida de la app
detector_worker = DetectorWorker(max_queue=30)
