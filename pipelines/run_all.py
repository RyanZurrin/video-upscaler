#!/usr/bin/env python3
# run_all.py
# Orchestrator: preprocess -> run 4 upscalers -> assemble videos -> build grid

import argparse
import subprocess
import os
from pathlib import Path
import shutil
import sys
import cv2

def call(cmd):
    print("CALL:", " ".join(cmd))
    subprocess.check_call(cmd)

def build_grid(original, realesr, swinir, svd, outpath):
    # scale each to same size and create 2x2 grid via ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-i", original, "-i", realesr, "-i", swinir, "-i", svd,
        "-filter_complex",
        "[0:v]scale=iw*2:ih*2[p0];"
        "[1:v]scale=iw*2:ih*2[p1];"
        "[2:v]scale=iw*2:ih*2[p2];"
        "[3:v]scale=iw*2:ih*2[p3];"
        "[p0][p1]hstack=inputs=2[top];"
        "[p2][p3]hstack=inputs=2[bottom];"
        "[top][bottom]vstack=inputs=2[grid]",
        "-map", "[grid]", "-c:v", "libx264", "-crf", "20", "-preset", "medium", outpath
    ]
    subprocess.check_call(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--realesr_ncnn", default=None, help="path to realesrgan-ncnn-vulkan.exe (optional)")
    p.add_argument("--realesr_model_dir", default="models/realesrgan")
    p.add_argument("--realesr_pth", default="models/realesrgan/realesr-general-x4v3.pth")
    p.add_argument("--swinir_pth", default="models/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth")
    p.add_argument("--bsrgan_pth", default="models/bsrgan/BSRGAN.pth")
    p.add_argument("--svd_model", default="stabilityai/stable-video-diffusion-img2vid-xt")
    p.add_argument("--denoise", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)
    tmp = outdir / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    frames_dir = tmp / "frames"

    # 1. Preprocess
    cmd = ["python", "preprocess_video.py", "--input", args.input, "--outdir", str(outdir)]
    if args.denoise:
        cmd += ["--denoise", "--median_k", "3"]
    call(cmd)

    # frames dir possibly in outdir/tmp/frames
    frames_dir = os.path.join(str(outdir), "tmp", "frames")
    if not os.path.exists(frames_dir):
        # fallback to outdir/tmp/frames naming difference
        frames_dir = os.path.join(str(outdir), "tmp", "frames")
    # get fps
    cap = cv2.VideoCapture(args.input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()

    # 2. Real-ESRGAN (NCNN or py fallback)
    realesr_frames = outdir / "out_realesr"
    realesr_frames.mkdir(exist_ok=True)
    cmd = ["python", "realesr_upscale.py", "--frames_dir", frames_dir, "--outdir", str(realesr_frames),
           "--ncnn_bin", args.realesr_ncnn or "" , "--ncnn_model_dir", args.realesr_model_dir,
           "--pth_model", args.realesr_pth]
    call([c for c in cmd if c != ""])

    # 3. SwinIR
    swinir_frames = outdir / "out_swinir"
    swinir_frames.mkdir(exist_ok=True)
    call(["python", "swinir_upscale.py", "--frames_dir", frames_dir, "--outdir", str(swinir_frames), "--model", args.swinir_pth, "--batch", "1"])

    # 4. BSRGAN
    bsrgan_frames = outdir / "out_bsrgan"
    bsrgan_frames.mkdir(exist_ok=True)
    call(["python", "bsrgan_upscale.py", "--frames_dir", frames_dir, "--outdir", str(bsrgan_frames), "--model", args.bsrgan_pth])

    # 5. Diffusion SR (slow)
    svd_frames = outdir / "out_svd"
    svd_frames.mkdir(exist_ok=True)
    call(["python", "svd_upscale.py", "--frames_dir", frames_dir, "--outdir", str(svd_frames), "--model", args.svd_model, "--steps", "12"])

    # 6. Reassemble each back to video
    from frames_to_video import frames_to_video
    realesr_vid = str(outdir / "realesr.mp4")
    swinir_vid = str(outdir / "swinir.mp4")
    bsrgan_vid = str(outdir / "bsrgan.mp4")
    svd_vid = str(outdir / "svd.mp4")
    orig_scaled = str(outdir / "original_scaled.mp4")
    frames_to_video(str(realesr_frames), realesr_vid, fps=fps, crf=18, preset="slow")
    frames_to_video(str(swinir_frames), swinir_vid, fps=fps, crf=18, preset="slow")
    frames_to_video(str(bsrgan_frames), bsrgan_vid, fps=fps, crf=18, preset="slow")
    frames_to_video(str(svd_frames), svd_vid, fps=fps, crf=18, preset="slow")
    call(["ffmpeg", "-y", "-i", args.input, "-vf", "scale=iw*2:ih*2", "-c:v", "libx264", "-crf", "20", "-preset", "medium", orig_scaled])

    # 7. Build comparison grid
    grid = str(outdir / "comparison_grid.mp4")
    build_grid(orig_scaled, realesr_vid, swinir_vid, svd_vid, grid)
    print("Finished. Comparison grid at:", grid)

if __name__ == "__main__":
    main()
