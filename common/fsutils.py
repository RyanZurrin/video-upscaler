import os
import shutil

def ensure_empty_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def list_frames(frames_dir, ext=".png"):
    return sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(ext)])
