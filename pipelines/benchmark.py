#!/usr/bin/env python3
"""
benchmark.py

Compute timing, GPU memory usage via nvidia-smi, and PSNR/SSIM between original frames and result frames.
Outputs CSV report.

Usage:
  python benchmark.py --orig_frames tmp/frames --result_frames results/realesrgan_frames --out report.csv
"""
import argparse
import csv
import os
import subprocess
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import cv2
from common.fsutils import list_frames

def gpu_memory_snapshot():
    try:
        out = subprocess.check_output(["nvidia-smi","--query-gpu=memory.used --format=csv,noheader,nounits"], encoding="utf-8")
        vals = [int(x.strip()) for x in out.strip().splitlines()]
        return vals
    except Exception:
        return []

def compare_pair(img1_path, img2_path):
    a = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    b = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    if a is None or b is None or a.shape != b.shape:
        return None, None
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    p = psnr(a, b, data_range=255)
    s = ssim(a, b, data_range=255)
    return p, s

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--orig_frames", required=True)
    p.add_argument("--result_frames", required=True)
    p.add_argument("--out", default="benchmark_report.csv")
    args = p.parse_args()

    orig = sorted([os.path.basename(x) for x in list_frames(args.orig_frames)])
    res = sorted([os.path.basename(x) for x in list_frames(args.result_frames)])
    common = sorted(list(set(orig).intersection(set(res))))
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame","psnr","ssim"])
        for fn in common:
            pval, sval = compare_pair(os.path.join(args.orig_frames, fn), os.path.join(args.result_frames, fn))
            if pval is None:
                continue
            writer.writerow([fn, f"{pval:.3f}", f"{sval:.4f}"])
    print("Wrote", args.out)
    print("GPU memory snapshot (MB):", gpu_memory_snapshot())

if __name__ == "__main__":
    main()
