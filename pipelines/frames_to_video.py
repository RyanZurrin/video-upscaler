#!/usr/bin/env python3
"""
frames_to_video.py
Assemble frames named 00000000.png etc into a video.
"""
import argparse
import os
from pathlib import Path
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    frames = Path(args.frames)
    pattern = os.path.join(str(frames), "%08d.png")
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-i", pattern, "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p", args.out]
    import subprocess
    subprocess.check_call(cmd)
if __name__ == "__main__":
    main()
