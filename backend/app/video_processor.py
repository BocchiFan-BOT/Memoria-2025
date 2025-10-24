# backend/app/video_processor.py
import cv2, threading, time
from collections import deque
import pandas as pd
import os, logging, traceback
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# --- Umbrales de alertas (con fallback seguro si no están en config) ---
try:
    from app.config import ALERT_COUNT_THRESHOLD, ALERT_OCC_THRESHOLD, ALERT_COOLDOWN_SEC
except Exception:
    ALERT_COUNT_THRESHOLD = 999999  # desactivado por defecto
    ALERT_OCC_THRESHOLD = 101.0     # desactivado por defecto (>%)
    ALERT_COOLDOWN_SEC = 30.0

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

def clamp_bbox(b, w, h):
    x1,y1,x2,y2 = b
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1,y1,x2,y2]

def xyxy_to_cxcywh(b):
    x1,y1,x2,y2 = b
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    cx = x1 + w/2.0
    cy = y1 + h/2.0
    return np.array([cx, cy, w, h], dtype=float)

def cxcywh_to_xyxy(s):
    cx, cy, w, h = s
    x1 = cx - w/2.0
    y1 = cy - h/2.0
    x2 = cx + w/2.0
    y2 = cy + h/2.0
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

def nms_xyxy(boxes: List[List[int]], iou_thr=0.55) -> List[List[int]]:
    if not boxes:
        return []
    boxes_np = np.array(boxes, dtype=float)
    x1,y1,x2,y2 = boxes_np[:,0], boxes_np[:,1], boxes_np[:,2], boxes_np[:,3]
    areas = (x2-x1+1)*(y2-y1+1)
    order = np.argsort(areas)[::-1]  # grandes primero (más estables)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w*h
        iou = inter/(areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds+1]
    return [boxes[i] for i in keep]

# ============================================================
# SORT-lite con Kalman + reglas anti-duplicado
# ============================================================
class KalmanBox:
    def __init__(self, init_state):
        self.x = np.zeros((8,1), dtype=float)
        self.x[:4,0] = init_state  # cx cy w h
        self.P = np.eye(8, dtype=float) * 10.0
        self.F = np.eye(8, dtype=float);  self.F[0,4]=self.F[1,5]=self.F[2,6]=self.F[3,7]=1.0
        self.H = np.zeros((4,8), dtype=float); self.H[0,0]=self.H[1,1]=self.H[2,2]=self.H[3,3]=1.0
        self.Q = np.eye(8, dtype=float) * 0.05
        self.R = np.eye(4, dtype=float) * 2.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = np.asarray(z, dtype=float).reshape(4,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(8)
        self.P = (I - K @ self.H) @ self.P

    def get_state_xyxy(self):
        return cxcywh_to_xyxy(self.x[:4,0])

class SortTrack:
    def __init__(self, tid: int, bbox_xyxy: List[int], max_age=20, min_hits=3, spawn_cooldown=3):
        self.id = tid
        self.kf = KalmanBox(xyxy_to_cxcywh(bbox_xyxy))
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.last_xyxy = bbox_xyxy
        self.max_age = max_age
        self.min_hits = min_hits
        self.is_dead = False
        self.last_seen_time = time.time()
        self.spawn_cooldown = spawn_cooldown  # frames iniciales protegidos

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.get_state_xyxy()

    def update(self, bbox_xyxy: List[int]):
        self.kf.update(xyxy_to_cxcywh(bbox_xyxy))
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_xyxy = bbox_xyxy
        self.last_seen_time = time.time()

    def mark_missed(self):
        self.hit_streak = 0
        if self.time_since_update > self.max_age:
            self.is_dead = True

    def xyxy(self):
        return self.kf.get_state_xyxy()

class SortTracker:
    def __init__(self, iou_thresh=0.3, max_age=20, min_hits=3, dup_merge_iou=0.8, init_iou_block=0.5, spawn_cooldown=3):
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.dup_merge_iou = float(dup_merge_iou)
        self.init_iou_block = float(init_iou_block)
        self.spawn_cooldown = int(spawn_cooldown)
        self.tracks: dict[int, SortTrack] = {}
        self.next_id = 1

    def _predictions(self):
        return {tid: tr.predict() for tid, tr in self.tracks.items()}

    def _associate_greedy(self, dets: List[List[int]], preds: dict):
        matched, unmatched_dets, unmatched_trks = [], set(range(len(dets))), set(self.tracks.keys())
        if len(dets) and len(preds):
            table = []
            for j, db in enumerate(dets):
                for tid, pb in preds.items():
                    iou = iou_xyxy(db, pb)
                    if iou >= self.iou_thresh:
                        table.append((iou, j, tid))
            table.sort(reverse=True)
            used_d, used_t = set(), set()
            for iou, j, tid in table:
                if j in used_d or tid in used_t:
                    continue
                matched.append((tid, j))
                used_d.add(j); used_t.add(tid)
            unmatched_dets -= used_d
            unmatched_trks -= used_t
        return matched, list(unmatched_dets), list(unmatched_trks)

    def _suppress_duplicates(self):
        tids = list(self.tracks.keys())
        to_remove = set()
        for i in range(len(tids)):
            for j in range(i+1, len(tids)):
                ti, tj = tids[i], tids[j]
                if ti in to_remove or tj in to_remove:
                    continue
                bi = self.tracks[ti].xyxy()
                bj = self.tracks[tj].xyxy()
                if iou_xyxy(bi, bj) >= self.dup_merge_iou:
                    keep = ti if self.tracks[ti].hits >= self.tracks[tj].hits else tj
                    drop = tj if keep == ti else ti
                    to_remove.add(drop)
        for tid in to_remove:
            self.tracks.pop(tid, None)

    def update(self, detections: List[List[int]], allow_new=True) -> List[Tuple[int, List[int]]]:
        preds = self._predictions()
        dets = detections

        matched, unmatched_dets, unmatched_trks = self._associate_greedy(dets, preds)

        for tid, j in matched:
            tr = self.tracks.get(tid)
            if tr:
                tr.update(dets[j])

        if allow_new:
            for j in unmatched_dets:
                db = dets[j]
                conflict = False
                for tid, pb in preds.items():
                    if iou_xyxy(db, pb) >= self.init_iou_block:
                        conflict = True; break
                if conflict:
                    continue
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = SortTrack(tid, db, max_age=self.max_age, min_hits=self.min_hits, spawn_cooldown=self.spawn_cooldown)

        for tid in unmatched_trks:
            tr = self.tracks.get(tid)
            if tr:
                tr.mark_missed()

        dead = [tid for tid, tr in self.tracks.items() if tr.is_dead]
        for tid in dead:
            self.tracks.pop(tid, None)
        self._suppress_duplicates()

        return [(tid, tr.xyxy()) for tid, tr in self.tracks.items()]

    def active_tracks_for_count(self, min_age_for_count=3, ghost_ttl=5) -> List[int]:
        now = time.time()
        eligible = []
        for tid, tr in self.tracks.items():
            alive = (tr.age >= min_age_for_count) or (tr.hit_streak >= self.min_hits) or (tr.spawn_cooldown > 0)
            recently_seen = (now - tr.last_seen_time) <= (ghost_ttl / 30.0)
            if alive or recently_seen:
                eligible.append(tid)
            if tr.spawn_cooldown > 0:
                tr.spawn_cooldown -= 1
        return eligible

# ----------------------------
# Suavizador de conteo
# ----------------------------
class CountStabilizer:
    def __init__(self, window=9, rate_limit_per_frame=3, clamp_min=0, clamp_max=9999):
        self.window = int(window)
        self.values = deque(maxlen=self.window)
        self.last_out = 0
        self.rate_limit = int(rate_limit_per_frame)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
    def push(self, v: int) -> int:
        self.values.append(int(v))
        med = int(np.median(self.values)) if self.values else 0
        if len(self.values) == 1:
            out = med
        else:
            delta = med - self.last_out
            if delta > self.rate_limit:   out = self.last_out + self.rate_limit
            elif delta < -self.rate_limit: out = self.last_out - self.rate_limit
            else:                          out = med
        out = max(self.clamp_min, min(self.clamp_max, out))
        self.last_out = out
        return out

# ----------------------------
# VideoProcessor
# ----------------------------
class VideoProcessor:
    def __init__(
        self,
        source,
        # --- Identificación de cámara (nuevo) ---
        cam_id: str = "",
        cam_name: str = "",
        # --- Salida / resize ---
        output_size=(1280, 720),
        letterbox_resize=True,
        # --- Historial / muestreo ---
        max_history=5000,
        metrics_interval_sec=5.0,
        aggregation_mode="peak",
        # --- Modelo / inferencia ---
        conf=0.30,
        imgsz=960,
        model_name="yolov8m-seg.pt",
        device=None,
        half=True,
        detect_interval=3,
        max_reuse_frames=30,
        # --- Tracker / conteo ---
        iou_thresh=0.3,
        max_age=20,
        min_hits=3,
        min_age_for_count=3,
        ghost_ttl=5,
        dup_merge_iou=0.8,
        init_iou_block=0.5,
        spawn_cooldown=3,
        # --- Visual / debug ---
        debug=False,
        overlay=False,
        overlay_alpha=0.35,
        # --- Smoothing ---
        ema_alpha=0.2,
        count_window=9,
        rate_limit_per_frame=3,
        # --- Rendimiento ---
        flush_grab=True,
        cache_cleanup_every=300
    ):
        self.source = source
        self.cam_id = cam_id
        self.cam_name = cam_name

        # RTSP baja latencia
        if isinstance(source, str) and source.lower().startswith("rtsp"):
            opts = "rtsp_transport;udp|max_delay;0|buffer_size;1024|stimeout;2000000|rw_timeout;2000000|reorder_queue_size;0|fpsprobesize;0|analyzeduration;0|max_analyze_duration;0|probesize;32|flags;low_delay|fflags;nobuffer"
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", opts)
            self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {source}")
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass

        # Torch / CUDA
        if TORCH_AVAILABLE:
            try:
                torch.backends.cudnn.benchmark = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            except Exception: pass

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
            logger.warning("ultralytics no instalado.")

        # Parámetros
        self.conf = float(conf); self.imgsz = int(imgsz)
        self.output_size = tuple(output_size)
        self.letterbox_resize = bool(letterbox_resize)
        self.detect_interval = max(1, int(detect_interval))
        self.max_reuse_frames = max(1, int(max_reuse_frames))
        self.debug = bool(debug); self.overlay = bool(overlay)
        self.overlay_alpha = float(overlay_alpha); self.ema_alpha = float(ema_alpha)
        self.metrics_interval_sec = float(metrics_interval_sec)
        self.aggregation_mode = aggregation_mode.lower()
        self.cache_cleanup_every = int(cache_cleanup_every)
        self.flush_grab = bool(flush_grab)
        self._last_cache_cleanup = time.time()

        # Tracking / conteo
        self.tracker = SortTracker(
            iou_thresh=iou_thresh, max_age=max_age, min_hits=min_hits,
            dup_merge_iou=dup_merge_iou, init_iou_block=init_iou_block,
            spawn_cooldown=spawn_cooldown
        )
        self.min_age_for_count = int(min_age_for_count)
        self.ghost_ttl = int(ghost_ttl)
        self.count_stab = CountStabilizer(window=int(count_window), rate_limit_per_frame=int(rate_limit_per_frame))

        if self.debug: os.makedirs("debug_frames", exist_ok=True)

        # Estado
        self.current_frame = None
        self.current_count_raw = 0
        self.current_count = 0
        self.current_crowd_index = 0.0
        self.crowd_ema = None
        self.count_history = deque(maxlen=max_history)
        self._stop = False

        # Alertas (nuevo)
        self.alerts: deque[Dict[str, Any]] = deque(maxlen=200)
        self._last_alert_ts = 0.0

        # Detección / flujo
        self._last_boxes = []
        self._frames_since_detect = 0
        self._frame_idx = 0
        self._prev_gray = None

        # Ventana muestreo
        self._window_samples = []
        self._window_start_t = time.time()

        # Warmup
        if self.model is not None:
            try:
                dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
                _ = self.model.predict(dummy, imgsz=self.imgsz, conf=self.conf,
                                       device=self.device, half=self.use_half, verbose=False)
                if _has_cuda(): torch.cuda.synchronize()
                logger.info("Warmup completado en %s (FP16=%s)", self.device, self.use_half)
            except Exception:
                logger.debug("Warmup falló: %s", traceback.format_exc())

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("VideoProcessor iniciado para: %s", source)

    # ----------------------------
    # Métricas
    # ----------------------------
    def _person_class_id(self):
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            for k, v in names.items():
                if str(v).lower() == "person": return int(k)
        elif isinstance(names, (list, tuple)):
            for i, v in enumerate(names):
                if str(v).lower() == "person": return int(i)
        return 0

    def _compute_crowd_index(self, res, frame_shape):
        try:
            h, w = frame_shape[:2]
            frame_area = float(h * w)
            boxes = getattr(res, "boxes", None)
            masks = getattr(res, "masks", None)
            if frame_area <= 0 or boxes is None or masks is None or getattr(masks, "data", None) is None:
                return 0.0
            pid = self._person_class_id()
            cls = getattr(boxes, "cls", None)
            if cls is None: return 0.0
            cls_cpu = cls.int().detach().cpu().numpy()
            person_mask = (cls_cpu == pid)
            if not np.any(person_mask): return 0.0
            sel = masks.data[person_mask]
            if sel.numel() == 0: return 0.0
            mask_area = float(sel.sum().item())
            Hm, Wm = sel.shape[-2], sel.shape[-1]
            if Hm>0 and Wm>0:
                return float(max(0.0, min(1.0, mask_area/float(Hm*Wm))))
            return float(mask_area/frame_area)
        except Exception:
            logger.debug("Error crowd index: %s", traceback.format_exc())
            return 0.0

    # ----------------------------
    # Propagación por Optical Flow (sólo corrige, no crea)
    # ----------------------------
    def _flow_update_boxes(self, prev_gray, gray, boxes):
        if len(boxes) == 0 or prev_gray is None:
            return boxes
        h, w = gray.shape[:2]
        new_boxes = []
        lk_params = dict(winSize=(21,21), maxLevel=3,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        for b in boxes:
            x1,y1,x2,y2 = b
            pts = np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2],[(x1+x2)//2,(y1+y2)//2]], dtype=np.float32).reshape(-1,1,2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **lk_params)
            if p1 is None or st is None: continue
            good_old = pts[st==1].reshape(-1,2)
            good_new = p1[st==1].reshape(-1,2)
            if len(good_old) < 2: continue
            shift = np.median(good_new - good_old, axis=0)
            dx, dy = float(shift[0]), float(shift[1])
            nb = [int(round(x1+dx)), int(round(y1+dy)), int(round(x2+dx)), int(round(y2+dy))]
            nb = clamp_bbox(nb, w, h)
            if nb is not None:
                new_boxes.append(nb)
        return new_boxes

    def _draw_overlays(self, frame, res, tracked: List[Tuple[int, List[int]]]):
        try:
            for tid, (x1, y1, x2, y2) in tracked:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"ID {tid}", (x1, max(12, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            masks = getattr(res, "masks", None)
            boxes = getattr(res, "boxes", None)
            if self.overlay and masks is not None and getattr(masks, "data", None) is not None and boxes is not None:
                mdata = masks.data
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
                    overlay[binmask == 1] = (overlay[binmask == 1]*(1 - self.overlay_alpha) + np.array((0,0,255))*self.overlay_alpha).astype(np.uint8)
                    frame[:] = overlay
        except Exception:
            logger.debug("Error overlay: %s", traceback.format_exc())

    def _aggregate_and_commit_window(self):
        if not self._window_samples: return
        arr = np.array(self._window_samples)
        t_values, counts, crowds = arr[:,0], arr[:,1], arr[:,2]
        if self.aggregation_mode == "peak":
            idx = int(np.argmax(crowds)); t_sel, c_sel, cr_sel = float(t_values[idx]), int(counts[idx]), float(crowds[idx])
        elif self.aggregation_mode == "mean":
            t_sel, c_sel, cr_sel = float(t_values[-1]), int(round(float(counts.mean()))), float(crowds.mean())
        elif self.aggregation_mode == "median":
            t_sel, c_sel, cr_sel = float(t_values[-1]), int(round(float(np.median(counts)))), float(np.median(crowds))
        else:
            idx = int(np.argmax(crowds)); t_sel, c_sel, cr_sel = float(t_values[idx]), int(counts[idx]), float(crowds[idx])
        self.count_history.append((t_sel, c_sel, cr_sel))
        self._window_samples.clear(); self._window_start_t = time.time()

    # ----------------------------
    # Bucle principal
    # ----------------------------
    def _run(self):
        while not self._stop:
            try:
                if self.flush_grab:
                    for _ in range(0): self.cap.grab()

                ret, frame = self.cap.read()
                if not ret or frame is None:
                    time.sleep(0.005);  continue

                if self.letterbox_resize:
                    frame, _, _ = letterbox(frame, self.output_size)
                else:
                    frame = cv2.resize(frame, self.output_size, interpolation=cv2.INTER_LINEAR)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                self._frame_idx += 1
                do_detect = (self._frame_idx % self.detect_interval == 0) or (self._frames_since_detect >= self.max_reuse_frames)

                raw_person_count = 0
                crowd_index = 0.0
                tracked = []
                res_for_draw = None

                if self.model and do_detect:
                    with torch.no_grad() if TORCH_AVAILABLE else _dummy_context():
                        results = self.model.predict(
                            frame[:, :, ::-1],
                            conf=self.conf,
                            imgsz=self.imgsz,
                            device=self.device,
                            half=self.use_half,
                            verbose=False
                        )
                    res = results[0]; res_for_draw = res

                    # Cajas person (+ NMS extra para evitar solapes residuales)
                    det_boxes = []
                    boxes = getattr(res, "boxes", None)
                    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
                        pid = self._person_class_id()
                        cls_cpu = boxes.cls.int().detach().cpu().numpy() if getattr(boxes, "cls", None) is not None else None
                        xyxy = boxes.xyxy.int().detach().cpu().numpy()
                        for i, bb in enumerate(xyxy):
                            if cls_cpu is None or cls_cpu[i] == pid:
                                x1,y1,x2,y2 = [int(v) for v in bb.tolist()]
                                det_boxes.append([x1,y1,x2,y2])
                    det_boxes = nms_xyxy(det_boxes, iou_thr=0.55)

                    tracked = self.tracker.update(det_boxes, allow_new=True)
                    self._last_boxes = det_boxes
                    self._frames_since_detect = 0

                    raw_person_count = len(self.tracker.active_tracks_for_count(
                        min_age_for_count=self.min_age_for_count, ghost_ttl=self.ghost_ttl
                    ))
                    crowd_index = self._compute_crowd_index(res, frame.shape)

                else:
                    propagated = self._flow_update_boxes(self._prev_gray, gray, self._last_boxes)
                    tracked = self.tracker.update(propagated, allow_new=False)
                    self._last_boxes = propagated
                    self._frames_since_detect += 1
                    crowd_index = float(self.crowd_ema) if self.crowd_ema is not None else 0.0
                    raw_person_count = len(self.tracker.active_tracks_for_count(
                        min_age_for_count=self.min_age_for_count, ghost_ttl=self.ghost_ttl
                    ))

                # EMA crowd
                if self.crowd_ema is None: self.crowd_ema = float(crowd_index)
                else: self.crowd_ema = self.ema_alpha*float(crowd_index) + (1-self.ema_alpha)*self.crowd_ema

                # Conteo estabilizado
                stable_count = self.count_stab.push(raw_person_count)

                # Dibujo
                if res_for_draw is not None:
                    self._draw_overlays(frame, res_for_draw, tracked)
                else:
                    for tid, (x1, y1, x2, y2) in tracked:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"ID {tid}", (x1, max(12, y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

                # HUD
                cv2.rectangle(frame, (10,10), (520,84), (0,0,0), -1)
                cv2.putText(frame, f"Personas: {stable_count}", (18,42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(frame, f"Aglomeracion: {self.crowd_ema:.2f}", (18,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                self.current_frame = frame
                self.current_count_raw = int(raw_person_count)
                self.current_count = int(stable_count)
                self.current_crowd_index = float(self.crowd_ema)

                # Ventana de muestreo
                now = time.time()
                self._window_samples.append((now, int(stable_count), float(self.crowd_ema)))
                if (now - self._window_start_t) >= self.metrics_interval_sec:
                    self._aggregate_and_commit_window()

                # --- Alertas (nuevo) ---
                try:
                    crowd_pct = float(self.crowd_ema) * 100.0  # 0..100
                    if (
                        (self.current_count >= ALERT_COUNT_THRESHOLD)
                        or (crowd_pct >= ALERT_OCC_THRESHOLD)
                    ) and (now - self._last_alert_ts >= float(ALERT_COOLDOWN_SEC)):
                        alert = {
                            "cam_id": self.cam_id or str(self.source),
                            "cam_name": self.cam_name or str(self.source),
                            "t": now,
                            "count": int(self.current_count),
                            "occupancy": round(crowd_pct, 1),
                        }
                        self.alerts.append(alert)
                        self._last_alert_ts = now
                except Exception:
                    logger.debug("Fallo generando alerta: %s", traceback.format_exc())

                self._prev_gray = gray
                time.sleep(0.001)

                # Limpieza CUDA
                if _has_cuda() and (time.time() - self._last_cache_cleanup) > self.cache_cleanup_every:
                    try:
                        torch.cuda.empty_cache()
                        self._last_cache_cleanup = time.time()
                    except Exception:
                        pass

            except Exception as e:
                logger.exception("Error en hilo VideoProcessor: %s", e)
                time.sleep(0.01)

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

    def get_current_count(self, stabilized=True):
        return self.current_count if stabilized else self.current_count_raw

    def get_current_crowd_index(self):
        return self.current_crowd_index

    def get_history(self):
        return [{'t': t, 'count': c, 'crowd_index': ci} for t, c, ci in list(self.count_history)]

    def export_csv(self, path):
        df = pd.DataFrame(self.get_history())
        df.to_csv(path, index=False)

    # --- NUEVO: API de alertas ---
    def get_alerts(self, since_ts: Optional[float] = None):
        try:
            if since_ts is None:
                return list(self.alerts)
            return [a for a in list(self.alerts) if float(a.get("t", 0)) >= float(since_ts)]
        except Exception:
            return []

    def stop(self):
        self._stop = True
        try: self._thread.join(timeout=1.0)
        except Exception: pass
        try: self.cap.release()
        except Exception: pass
        if _has_cuda():
            try: torch.cuda.empty_cache()
            except Exception: pass
        logger.info("VideoProcessor detenido: %s", self.source)

class _dummy_context:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
