# backend/app/video_processor.py
import cv2
import threading
import time
from collections import deque
from ultralytics import YOLO
import torch

class VideoProcessor:
    def __init__(self, path, max_history=1000, conf=0.5, imgsz=640):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video/cámara: {path}")

        # Modelo YOLOv8s (preciso y razonable en CPU)
        self.model = YOLO("yolov8s.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf = conf
        self.imgsz = imgsz

        self.count_history = deque(maxlen=max_history)  # (timestamp, count)
        self.current_frame = None
        self.current_count = 0

        self._stop = False
        self._thread = threading.Thread(target=self._update_frames, daemon=True)
        self._thread.start()

    def _update_frames(self):
        # Opcional: reduce resolución para más FPS en CPU
        target_w = 960  # ajusta si quieres más/menos performance
        while not self._stop:
            ret, frame = self.cap.read()
            if not ret:
                # reinicia si es un archivo
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize para acelerar inferencia en CPU
            h, w = frame.shape[:2]
            if w > target_w:
                ratio = target_w / w
                frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)

            # Inferencia YOLO (sólo personas)
            # results[0] es el primer resultado; boxes incluye xyxy, cls y conf
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=0.45,
                device=self.device,
                verbose=False
            )
            r = results[0]

            person_count = 0
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names.get(cls_id, str(cls_id))
                    if cls_name != "person":
                        continue
                    person_count += 1
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    conf = float(box.conf[0])

                    # Caja roja y etiqueta
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"person {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 255), -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

            # Overlay con conteo total
            cv2.rectangle(frame, (10, 10), (220, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"Personas: {person_count}", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

            self.current_frame = frame
            self.current_count = person_count
            self.count_history.append((time.time(), person_count))

            # Pequeño sleep evita 100% CPU si el video es corto
            time.sleep(0.01)

    def get_frame(self):
        if self.current_frame is None:
            return None
        ret, jpeg = cv2.imencode('.jpg', self.current_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            return None
        return jpeg.tobytes()

    def get_current_count(self):
        return self.current_count

    def get_history(self):
        return [{'t': t, 'count': c} for t, c in list(self.count_history)]

    def export_csv(self, path):
        import pandas as pd
        df = pd.DataFrame(self.get_history())
        df.to_csv(path, index=False)

    def stop(self):
        self._stop = True
        try:
            self._thread.join(timeout=1.0)
        except:
            pass
        if self.cap:
            self.cap.release()

