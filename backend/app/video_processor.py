# backend/app/video_processor.py
import cv2, threading, time
from collections import deque
import pandas as pd

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class VideoProcessor:
    def __init__(self, source, max_history=1000, conf=0.4, imgsz=640):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {source}")

        self.model = YOLO("yolov8s.pt") if YOLO_AVAILABLE else None
        self.conf = conf
        self.imgsz = imgsz
        self.current_frame = None
        self.current_count = 0
        self.count_history = deque(maxlen=max_history)
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.5)
                continue

            # resize si es muy ancho
            h, w = frame.shape[:2]
            if w > 960:
                r = 960 / w
                frame = cv2.resize(frame, (int(w * r), int(h * r)))

            person_count = 0

            if self.model:
                # 🚀 forma moderna de inferencia
                results = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)

                for r in results:
                    if hasattr(r, "boxes") and r.boxes is not None:
                        for box in r.boxes:
                            cls = int(box.cls.item())
                            conf = float(box.conf.item())
                            name = self.model.names[cls]

                            if name == "person":
                                person_count += 1
                                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f"person {conf:.2f}",
                                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (255, 255, 255), 1)

            # overlay con conteo
            cv2.rectangle(frame, (10, 10), (240, 50), (0, 0, 0), -1)
            cv2.putText(frame, f"Personas detectadas: {person_count}",
                        (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.current_frame = frame
            self.current_count = person_count
            self.count_history.append((time.time(), person_count))
            time.sleep(0.03)

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


