#!/usr/bin/env python3
# preprocess_video.py
# Extract frames, optional temporal median denoise, and simple scene detection.
import os
import cv2
import argparse
import json
from tqdm import tqdm
import numpy as np

def extract_frames(infile, outdir):
    os.makedirs(outdir, exist_ok=True)
    cap = cv2.VideoCapture(infile)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = 0
    pbar = tqdm(total=total, desc="Extract frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fname = os.path.join(outdir, f"{idx:08d}.png")
        cv2.imwrite(fname, frame)
        idx += 1
        pbar.update(1)
    cap.release()
    pbar.close()
    return idx

def median_temporal_denoise(frames_dir, outdir, k=3):
    files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
    n = len(files)
    pad = k // 2
    os.makedirs(outdir, exist_ok=True)
    pbar = tqdm(total=n, desc="Temporal median denoise")
    # Read in sliding manner to save memory
    imgs = [None]*n
    for i, f in enumerate(files):
        imgs[i] = cv2.imread(os.path.join(frames_dir, f))
    for i in range(n):
        window = []
        for j in range(i-pad, i+pad+1):
            jj = min(max(j, 0), n-1)
            window.append(imgs[jj].astype(np.int16))
        med = np.median(np.stack(window, axis=0), axis=0).astype(np.uint8)
        cv2.imwrite(os.path.join(outdir, files[i]), med)
        pbar.update(1)
    pbar.close()
    return outdir

def simple_scene_detect(frames_dir, threshold=30.0):
    files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".png")])
    if not files:
        return []
    prev = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE).astype(np.float32)
    segs = []
    start = 0
    for i, f in enumerate(files[1:], start=1):
        cur = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        diff = float(np.mean(np.abs(cur - prev)))
        if diff > threshold:
            segs.append((start, i-1))
            start = i
        prev = cur
    segs.append((start, len(files)-1))
    return segs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--denoise", action="store_true")
    p.add_argument("--median_k", type=int, default=3)
    p.add_argument("--scene_detect", action="store_true")
    p.add_argument("--scene_thresh", type=float, default=30.0)
    args = p.parse_args()

    tmp = os.path.join(args.outdir, "tmp")
    frames = os.path.join(tmp, "frames")
    os.makedirs(tmp, exist_ok=True)
    n = extract_frames(args.input, frames)
    print(f"Extracted {n} frames -> {frames}")
    processed_frames = frames
    if args.denoise:
        denoised = os.path.join(tmp, "denoised")
        processed_frames = median_temporal_denoise(frames, denoised, k=args.median_k)
        print("Denoised frames ->", processed_frames)
    segments = []
    if args.scene_detect:
        segments = simple_scene_detect(processed_frames, threshold=args.scene_thresh)
        with open(os.path.join(tmp, "segments.json"), "w") as fh:
            json.dump(segments, fh)
        print("Segments written to", os.path.join(tmp, "segments.json"))
    print("Preprocess complete. frames_dir =", processed_frames)

if __name__ == "__main__":
    main()
