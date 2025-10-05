# backend/app/video_processor.py
import cv2, threading, time
from collections import deque
import pandas as pd
import os, logging, traceback

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VideoProcessor")

class VideoProcessor:
    def __init__(self, source, max_history=1000, conf=0.25, imgsz=640, debug=False):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {source}")

        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO("yolov8s.pt")
                logger.info("YOLO cargado.")
            except Exception:
                logger.exception("Error cargando YOLO")
                self.model = None
        else:
            logger.warning("ultralytics no instalado: detección deshabilitada.")

        self.conf = conf
        self.imgsz = imgsz
        self.current_frame = None
        self.current_count = 0
        self.count_history = deque(maxlen=max_history)
        self._stop = False
        self.debug = debug
        if self.debug:
            os.makedirs("debug_frames", exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("VideoProcessor iniciado para: %s", source)

    def _run(self):
        while not self._stop:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.5)
                    continue

                # resize para no saturar CPU si imagen muy grande
                h, w = frame.shape[:2]
                if w > 1280:
                    scale = 1280 / w
                    frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

                person_count = 0

                if self.model:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.model(rgb, conf=self.conf, imgsz=self.imgsz, verbose=False)
                    for res in results:
                        boxes = getattr(res, "boxes", None)
                        if boxes is None:
                            continue
                        for box in boxes:
                            try:
                                cls_id = int(box.cls.item()) if hasattr(box.cls, 'item') else int(box.cls)
                                conf_score = float(box.conf.item()) if hasattr(box.conf, 'item') else float(box.conf)
                                name = self.model.names.get(cls_id, str(cls_id)) if hasattr(self.model, "names") else str(cls_id)
                                if "person" in str(name).lower():
                                    person_count += 1
                                    xy = box.xyxy[0]
                                    x1, y1, x2, y2 = [int(v) for v in xy.int().tolist()]
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                    cv2.putText(frame, f"P {conf_score:.2f}", (x1, max(12, y1-6)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                            except Exception:
                                logger.debug("Error procesando caja: %s", traceback.format_exc())

                # overlay conteo
                cv2.rectangle(frame, (10,10), (300,48), (0,0,0), -1)
                cv2.putText(frame, f"Personas: {person_count}", (18,36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                self.current_frame = frame
                self.current_count = person_count
                self.count_history.append((time.time(), person_count))

                if self.debug and person_count > 0:
                    fname = f"debug_frames/{int(time.time())}_{person_count}.jpg"
                    cv2.imwrite(fname, frame)

                time.sleep(0.02)

            except Exception as e:
                logger.exception("Error en hilo VideoProcessor: %s", e)
                time.sleep(1.0)

    def get_frame(self):
        if self.current_frame is None:
            return None
        try:
            ret, jpeg = cv2.imencode('.jpg', self.current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret: return None
            return jpeg.tobytes()
        except Exception:
            logger.exception("Error encoding JPEG")
            return None

    def get_current_count(self):
        return self.current_count

    def get_history(self):
        return [{'t': t, 'count': c} for t,c in list(self.count_history)]

    def export_csv(self, path):
        df = pd.DataFrame(self.get_history())
        df.to_csv(path, index=False)

    def stop(self):
        self._stop = True
        try:
            self._thread.join(timeout=1.0)
        except:
            pass
        try:
            self.cap.release()
        except:
            pass
        logger.info("VideoProcessor detenido: %s", self.source)


