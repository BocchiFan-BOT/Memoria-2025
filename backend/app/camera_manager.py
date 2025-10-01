# backend/app/camera_manager.py
import threading, json, os, time
from .video_processor import VideoProcessor

CAMERAS_FILE = os.path.join(os.path.dirname(__file__), "static", "cameras.json")
_lock = threading.Lock()

# in-memory cameras list: dict id -> {id,name,url,location,coordinates}
_cameras = {}
_processors = {}

def load_from_disk():
    try:
        if os.path.exists(CAMERAS_FILE):
            with open(CAMERAS_FILE, "r", encoding="utf-8") as f:
                arr = json.load(f)
                for c in arr:
                    _cameras[c['id']] = c
    except Exception:
        pass

def persist_to_disk():
    try:
        os.makedirs(os.path.dirname(CAMERAS_FILE), exist_ok=True)
        with open(CAMERAS_FILE, "w", encoding="utf-8") as f:
            json.dump(list(_cameras.values()), f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def list_cameras():
    with _lock:
        return list(_cameras.values())

def add_camera(cam):
    with _lock:
        _cameras[cam['id']] = cam
        persist_to_disk()

def remove_camera(cam_id):
    with _lock:
        if cam_id in _cameras:
            _cameras.pop(cam_id)
        if cam_id in _processors:
            try:
                _processors[cam_id].stop()
            except:
                pass
            _processors.pop(cam_id)
        persist_to_disk()

def get_camera(cam_id):
    return _cameras.get(cam_id)

def get_processor(cam_id, create_if_missing=True):
    with _lock:
        if cam_id in _processors:
            return _processors[cam_id]
        cam = _cameras.get(cam_id)
        if not cam or not create_if_missing:
            return None
        try:
            vp = VideoProcessor(source=cam['url'])
            _processors[cam_id] = vp
            return vp
        except Exception:
            return None

# Load cameras on import
load_from_disk()
