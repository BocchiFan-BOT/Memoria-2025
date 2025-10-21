# backend/app/video_processor.py
import cv2, threading, time
from collections import deque
import pandas as pd
import os, logging, traceback
import numpy as np
from typing import List, Tuple, Optional

# --- GPU / Torch ---
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VideoProcessor")

def _has_cuda():
    return TORCH_AVAILABLE and torch.cuda.is_available()

# ----------------------------
# Utilidades de imagen
# ----------------------------
def letterbox(im, new_shape=(1280, 720), color=(114, 114, 114)):
    """
    Redimensiona con padding (letterbox) manteniendo aspecto.
    Devuelve: imagen, escala, padding (dw, dh)
    """
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h0, w0 = shape
    w, h = new_shape

    scale = min(w / w0, h / h0)
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))

    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw, dh = (w - nw) / 2, (h - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im_padded = cv2.copyMakeBorder(im_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im_padded, scale, (left, top)

def iou_xyxy(a, b):
    """
    IoU entre dos cajas [x1,y1,x2,y2]
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)

# ----------------------------
# Tracker muy ligero por IoU
# ----------------------------
class IoUTracker:
    def __init__(self, iou_thresh=0.3, max_age=15):
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self.next_id = 1
        self.tracks = {}   # id -> {"bbox": [x1,y1,x2,y2], "age":0, "last_seen":t}

    def update(self, detections: List[List[int]]) -> List[Tuple[int, List[int]]]:
        """
        detections: lista de bboxes [x1,y1,x2,y2]
        retorna: lista de (id, bbox)
        """
        t = time.time()
        assigned = set()
        track_ids = list(self.tracks.keys())
        # Intento de asociación greedy
        for tid in track_ids:
            tb = self.tracks[tid]["bbox"]
            # hallar mejor match por IoU
            best_iou, best_j = 0.0, -1
            for j, db in enumerate(detections):
                if j in assigned:
                    continue
                iou = iou_xyxy(tb, db)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= self.iou_thresh and best_j >= 0:
                # asociar
                self.tracks[tid]["bbox"] = detections[best_j]
                self.tracks[tid]["age"] = 0
                self.tracks[tid]["last_seen"] = t
                assigned.add(best_j)
            else:
                self.tracks[tid]["age"] += 1

        # Nuevas pistas
        for j, db in enumerate(detections):
            if j in assigned:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": db, "age": 0, "last_seen": t}

        # Remover viejas
        dead = [tid for tid, tr in self.tracks.items() if tr["age"] > self.max_age]
        for tid in dead:
            self.tracks.pop(tid, None)

        return [(tid, tr["bbox"]) for tid, tr in self.tracks.items()]

# ----------------------------
# VideoProcessor
# ----------------------------
class VideoProcessor:
    """
    - Estandariza resolución (letterbox a output_size)
    - Detección + segmentación (YOLOv8-seg)
    - Conteo de personas
    - Índice de aglomeración = área segmentada / área del frame (0..1), con EMA
    - Tracking ligero por IoU (IDs persistentes)
    - Registro por intervalos (metrics_interval_sec) con selección de muestra "mejor" de la ventana
    - Optimizado para GPU (CUDA, FP16, warmup, no_grad, cudnn.benchmark)
    """
    def __init__(
        self,
        source,
        # --- Estandarización de salida ---
        output_size=(1280, 720),      # resolución estándar para todas las cámaras
        letterbox_resize=True,
        # --- Historial / muestreo ---
        max_history=5000,
        metrics_interval_sec=2.0,     # cada cuántos segundos guardar una fila al historial/DF
        aggregation_mode="peak",      # "peak" | "mean" | "median"
        # --- Modelo / inferencia ---
        conf=0.30,
        imgsz=960,
        model_name="yolov8l-seg.pt",
        device=None,                  # "cuda" | "cpu" | None (auto)
        half=True,
        detect_interval=1,            # ejecutar detección cada N frames (>=1). 1 = cada frame
        # --- Visual / debug ---
        debug=False,
        overlay=True,
        overlay_alpha=0.35,
        # --- Smoothing ---
        ema_alpha=0.2,
        # --- Rendimiento / mantenimiento ---
        cache_cleanup_every=300
    ):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {source}")

        # Buffer bajo para streaming RTSP/USB
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass

        # Torch / CUDA
        if TORCH_AVAILABLE:
            try:
                torch.backends.cudnn.benchmark = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        self.device = device or ("cuda" if _has_cuda() else "cpu")
        self.use_half = bool(half and self.device == "cuda")

        # Modelo
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)
                if hasattr(self.model, "fuse"):
                    self.model.fuse()
                logger.info("Modelo YOLO cargado: %s", model_name)
            except Exception:
                logger.exception("Error cargando YOLO")
                self.model = None
        else:
            logger.warning("ultralytics no instalado: detección/segmentación deshabilitada.")

        # Parámetros
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.output_size = tuple(output_size)
        self.letterbox_resize = bool(letterbox_resize)
        self.detect_interval = max(1, int(detect_interval))
        self.debug = bool(debug)
        self.overlay = bool(overlay)
        self.overlay_alpha = float(overlay_alpha)
        self.ema_alpha = float(ema_alpha)
        self.metrics_interval_sec = float(metrics_interval_sec)
        self.aggregation_mode = aggregation_mode.lower()
        self.cache_cleanup_every = int(cache_cleanup_every)
        self._last_cache_cleanup = time.time()

        if self.debug:
            os.makedirs("debug_frames", exist_ok=True)

        # Estado
        self.current_frame = None
        self.current_count = 0
        self.current_crowd_index = 0.0
        self.crowd_ema = None
        self.count_history = deque(maxlen=max_history)  # [(t, count, crowd)]
        self._stop = False

        # Tracker
        self.tracker = IoUTracker(iou_thresh=0.3, max_age=15)
        self._last_boxes = []  # bboxes de la última inferencia (para reciclar en frames saltados)
        self._frame_idx = 0

        # Ventana para muestreo temporal
        self._window_samples = []  # [(t, count, crowd), ...]
        self._window_start_t = time.time()

        # Warmup
        if self.model is not None:
            try:
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                _ = self.model.predict(
                    dummy,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    device=self.device,
                    half=self.use_half,
                    verbose=False
                )
                if _has_cuda():
                    torch.cuda.synchronize()
                logger.info("Warmup completado en %s (FP16=%s)", self.device, self.use_half)
            except Exception:
                logger.debug("Warmup falló: %s", traceback.format_exc())

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("VideoProcessor iniciado para: %s", source)

    # ----------------------------
    # Cálculo de métricas
    # ----------------------------
    def _person_class_id(self):
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == "person":
                    return int(k)
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                if str(v).lower() == "person":
                    return int(i)
        return 0

    def _compute_crowd_index(self, res, frame_shape):
        try:
            h, w = frame_shape[:2]
            frame_area = float(h * w)
            if frame_area <= 0:
                return 0.0

            boxes = getattr(res, "boxes", None)
            masks = getattr(res, "masks", None)
            if boxes is None or masks is None or getattr(masks, "data", None) is None:
                return 0.0

            pid = self._person_class_id()
            cls = getattr(boxes, "cls", None)
            if cls is None:
                return 0.0

            cls_cpu = cls.int().detach().cpu().numpy()
            person_mask = (cls_cpu == pid)
            if not np.any(person_mask):
                return 0.0

            sel = masks.data[person_mask]
            if sel.numel() == 0:
                return 0.0

            mask_area = float(sel.sum().item())
            Hm, Wm = sel.shape[-2], sel.shape[-1]
            if Hm > 0 and Wm > 0 and (Hm != h or Wm != w):
                norm_frac = mask_area / float(Hm * Wm)
                crowd_index = norm_frac
            else:
                crowd_index = mask_area / frame_area

            return float(max(0.0, min(1.0, crowd_index)))
        except Exception:
            logger.debug("Error calculando crowd index: %s", traceback.format_exc())
            return 0.0

    def _draw_overlays(self, frame, res, tracked: List[Tuple[int, List[int]]]):
        try:
            # cajas + ids
            for tid, (x1, y1, x2, y2) in tracked:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(12, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            # máscaras (opcional)
            masks = getattr(res, "masks", None)
            boxes = getattr(res, "boxes", None)
            if self.overlay and masks is not None and getattr(masks, "data", None) is not None and boxes is not None:
                mdata = masks.data  # [N,H,W]
                boxes_cls = boxes.cls.int().detach().cpu().numpy()
                pid = self._person_class_id()
                person_mask = (boxes_cls == pid)
                if np.any(person_mask):
                    sel = mdata[person_mask].detach().cpu().numpy()
                    binmask = (sel > 0.5).any(axis=0).astype(np.uint8)
                    Hm, Wm = binmask.shape
                    Hf, Wf = frame.shape[:2]
                    if (Hm != Hf) or (Wm != Wf):
                        binmask = cv2.resize(binmask, (Wf, Hf), interpolation=cv2.INTER_NEAREST)
                    overlay = frame.copy()
                    overlay[binmask == 1] = (overlay[binmask == 1] * (1 - self.overlay_alpha) + np.array((0,0,255)) * self.overlay_alpha).astype(np.uint8)
                    frame[:] = overlay
        except Exception:
            logger.debug("Error dibujando overlay: %s", traceback.format_exc())

    def _aggregate_and_commit_window(self):
        """
        Selecciona una muestra de la ventana y la agrega al historial.
        - peak: mayor crowd_index
        - mean/median: promedio/mediana de count y crowd_index
        """
        if not self._window_samples:
            return
        arr = np.array(self._window_samples)  # shape [N, 3] -> t, count, crowd
        t_values = arr[:,0]
        counts = arr[:,1]
        crowds = arr[:,2]

        if self.aggregation_mode == "peak":
            idx = int(np.argmax(crowds))
            t_sel = float(t_values[idx]); c_sel = int(counts[idx]); cr_sel = float(crowds[idx])
        elif self.aggregation_mode == "mean":
            t_sel = float(t_values[-1])  # timestamp del fin de ventana
            c_sel = int(round(float(counts.mean())))
            cr_sel = float(crowds.mean())
        elif self.aggregation_mode == "median":
            t_sel = float(t_values[-1])
            c_sel = int(round(float(np.median(counts))))
            cr_sel = float(np.median(crowds))
        else:
            # fallback -> peak
            idx = int(np.argmax(crowds))
            t_sel = float(t_values[idx]); c_sel = int(counts[idx]); cr_sel = float(crowds[idx])

        self.count_history.append((t_sel, c_sel, cr_sel))
        self._window_samples.clear()
        self._window_start_t = time.time()

    # ----------------------------
    # Bucle principal
    # ----------------------------
    def _run(self):
        while not self._stop:
            try:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.5)
                    continue

                # Estandariza resolución para TODAS las cámaras
                if self.letterbox_resize:
                    frame, _, _ = letterbox(frame, self.output_size)
                else:
                    frame = cv2.resize(frame, self.output_size, interpolation=cv2.INTER_LINEAR)

                self._frame_idx += 1
                do_detect = (self._frame_idx % self.detect_interval == 0)

                person_count = 0
                crowd_index = 0.0
                tracked = []

                # Detección/segmentación (según intervalo)
                if self.model and do_detect:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with torch.no_grad() if TORCH_AVAILABLE else _dummy_context():
                        results = self.model.predict(
                            rgb,
                            conf=self.conf,
                            imgsz=self.imgsz,
                            device=self.device,
                            half=self.use_half,
                            verbose=False
                        )
                    # Tomamos primer resultado
                    res = results[0]

                    # Extraer cajas de personas
                    boxes = getattr(res, "boxes", None)
                    det_boxes = []
                    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                        pid = self._person_class_id()
                        cls_cpu = boxes.cls.int().detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else None
                        xyxy = boxes.xyxy.int().detach().cpu().numpy()
                        for i, bb in enumerate(xyxy):
                            if cls_cpu is None or cls_cpu[i] == pid:
                                x1,y1,x2,y2 = [int(v) for v in bb.tolist()]
                                det_boxes.append([x1,y1,x2,y2])

                    self._last_boxes = det_boxes

                    # Conteo
                    person_count = len(det_boxes)
                    # Crowd index del resultado
                    crowd_index = self._compute_crowd_index(res, frame.shape)

                    # Tracking (con cajas recién detectadas)
                    tracked = self.tracker.update(det_boxes)

                    # Dibujos
                    self._draw_overlays(frame, res, tracked)

                else:
                    # No detectamos en este frame: reciclamos las últimas cajas para mantener dibujo/IDs
                    det_boxes = self._last_boxes
                    person_count = len(det_boxes)
                    tracked = self.tracker.update(det_boxes)

                    # Dibujo básico (sin máscaras nuevas)
                    for tid, (x1,y1,x2,y2) in tracked:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"ID {tid}", (x1, max(12, y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

                    # crowd_index se mantiene (sin nueva segmentación): usa último EMA
                    crowd_index = float(self.crowd_ema) if self.crowd_ema is not None else 0.0

                # EMA para crowd_index
                if self.crowd_ema is None:
                    self.crowd_ema = float(crowd_index)
                else:
                    self.crowd_ema = self.ema_alpha * float(crowd_index) + (1 - self.ema_alpha) * self.crowd_ema

                # Overlay de métricas en pantalla
                cv2.rectangle(frame, (10,10), (480,76), (0,0,0), -1)
                cv2.putText(frame, f"Personas: {person_count}", (18,42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(frame, f"Aglomeracion: {self.crowd_ema:.2f}", (18,68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                self.current_frame = frame
                self.current_count = person_count
                self.current_crowd_index = float(self.crowd_ema)

                # Muestreo por ventana temporal: acumular
                now = time.time()
                self._window_samples.append((now, person_count, float(self.crowd_ema)))
                if (now - self._window_start_t) >= self.metrics_interval_sec:
                    self._aggregate_and_commit_window()

                # Descanso pequeño (evita busy-wait)
                time.sleep(0.002)

                # Limpieza periódica caché CUDA
                if _has_cuda() and (time.time() - self._last_cache_cleanup) > self.cache_cleanup_every:
                    try:
                        torch.cuda.empty_cache()
                        self._last_cache_cleanup = time.time()
                    except Exception:
                        pass

            except Exception as e:
                logger.exception("Error en hilo VideoProcessor: %s", e)
                time.sleep(0.02)

    # ----------------------------
    # API pública
    # ----------------------------
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

    def get_current_crowd_index(self):
        return self.current_crowd_index

    def get_history(self):
        # [{'t':..., 'count':..., 'crowd_index':...}, ...]
        return [{'t': t, 'count': c, 'crowd_index': ci} for t, c, ci in list(self.count_history)]

    def export_csv(self, path):
        df = pd.DataFrame(self.get_history())
        df.to_csv(path, index=False)

    def stop(self):
        self._stop = True
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass
        if _has_cuda():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("VideoProcessor detenido: %s", self.source)

# Context manager dummy si no hay torch
class _dummy_context:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
