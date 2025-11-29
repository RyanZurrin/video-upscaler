#!/usr/bin/env python3
# frames_to_video.py
import argparse
import subprocess
import os

def frames_to_video(frames_dir, out_video, fps=30, crf=18, preset="slow"):
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%08d.png"),
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p", out_video
    ]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--preset", default="slow")
    args = p.parse_args()
    frames_to_video(args.frames_dir, args.out, fps=args.fps, crf=args.crf, preset=args.preset)
