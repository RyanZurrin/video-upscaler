#!/usr/bin/env python3
"""
run_all.py

High level orchestrator. Uses other scripts in the folder.
All paths are CLI args or discovered from common.paths.
"""
import argparse
import os
import subprocess
from pathlib import Path
from common.paths import INPUT_DIR, MODELS_DIR, OUTPUT_DIR, TMP_DIR
from common.fsutils import ensure_empty_dir, list_frames

def run_preprocess(inp, tmp, denoise=False, median_k=3):
    cmd = ["python", "pipelines/preprocess_video.py", "--input", inp, "--outdir", tmp]
    if denoise:
        cmd += ["--denoise", "--median_k", str(median_k)]
    subprocess.check_call(cmd)

def run_realesrgan_bin(bin_path, frames_dir, out_frames_dir, model_name):
    # many ncnn builds accept folder mode; if not, fallback to per-file loop
    out_frames_dir = str(out_frames_dir)
    os.makedirs(out_frames_dir, exist_ok=True)
    cmd = [bin_path, "-i", frames_dir, "-o", out_frames_dir, "-n", model_name]
    subprocess.check_call(cmd)

def frames_to_video(frames_dir, out_video, fps):
    cmd = ["python", "pipelines/frames_to_video.py", "--frames", frames_dir, "--out", out_video, "--fps", str(int(fps))]
    subprocess.check_call(cmd)

def build_grid(outdir, orig_scaled, realesr, swinir, svd):
    grid_path = os.path.join(outdir, "comparison_grid.mp4")
    # call ffmpeg complex. Keep it simple: hstack and vstack
    cmd = [
      "ffmpeg", "-y", "-i", orig_scaled, "-i", realesr, "-i", swinir, "-i", svd,
      "-filter_complex",
      "[0:v]scale=iw:ih[p0];[1:v]scale=iw:ih[p1];[2:v]scale=iw:ih[p2];[3:v]scale=iw:ih[p3];"
      "[p0][p1]hstack=inputs=2[top];[p2][p3]hstack=inputs=2[bottom];[top][bottom]vstack=inputs=2[grid]",
      "-map", "[grid]", "-c:v", "libx264", "-crf", "20", "-preset", "medium", grid_path
    ]
    subprocess.check_call(cmd)
    print("Grid saved to", grid_path)
    return grid_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="input video file")
    p.add_argument("--outdir", default=OUTPUT_DIR)
    p.add_argument("--realesrgan_bin", default="", help="path to realesrgan-ncnn-vulkan.exe")
    p.add_argument("--realesrgan_model", default="", help="name of model folder under models/realesrgan or path to model folder")
    p.add_argument("--swinir_model", default=os.path.join(MODELS_DIR, "swinir"))
    p.add_argument("--svd_model", default="stabilityai/stable-video-diffusion-img2vid-xt")
    p.add_argument("--denoise", action="store_true")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(TMP_DIR)
    ensure_empty_dir(str(tmp))

    # 1) preprocess
    run_preprocess(args.input, str(tmp), denoise=args.denoise)

    frames_dir = tmp / "denoised" if args.denoise else tmp / "frames"
    fps = float(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=r_frame_rate","-of","default=noprint_wrappers=1:nokey=1", args.input]).decode().strip().split('/')[0]) or 30

    # 2) Real-ESRGAN
    reales_out = outdir / "realesrgan_frames"
    ensure_empty_dir(str(reales_out))
    if args.realesrgan_bin:
        model_name = os.path.basename(args.realesrgan_model.rstrip("/\\"))
        run_realesrgan_bin(args.realesrgan_bin, str(frames_dir), str(reales_out), model_name)
    else:
        print("No realesrgan binary specified; skipping.")

    # 3) SwinIR
    swinir_out = outdir / "swinir_frames"
    ensure_empty_dir(str(swinir_out))
    subprocess.check_call(["python","pipelines/swinir_upscale.py","--frames_dir", str(frames_dir), "--outdir", str(swinir_out),"--model", args.swinir_model, "--batch","1"])

    # 4) SVD diffusion
    svd_out = outdir / "svd_frames"
    ensure_empty_dir(str(svd_out))
    subprocess.check_call(["python","pipelines/svd_upscale.py","--frames_dir", str(frames_dir), "--outdir", str(svd_out), "--model", args.svd_model, "--steps","12"])

    # 5) frames -> videos
    orig_scaled = os.path.join(outdir, "original_scaled.mp4")
    subprocess.check_call(["ffmpeg","-y","-i", args.input, "-vf", "scale=iw*2:ih*2", "-c:v","libx264","-crf","20","-preset","medium", orig_scaled])

    r_vid = os.path.join(outdir, "realesrgan.mp4")
    s_vid = os.path.join(outdir, "swinir.mp4")
    d_vid = os.path.join(outdir, "svd.mp4")
    frames_to_video(str(reales_out), r_vid, fps)
    frames_to_video(str(swinir_out), s_vid, fps)
    frames_to_video(str(svd_out), d_vid, fps)

    # 6) build grid and copy compare.html to outdir
    grid = build_grid(outdir, orig_scaled, r_vid, s_vid, d_vid)
    import shutil
    shutil.copy(os.path.join(os.path.dirname(__file__), "compare.html"), os.path.join(outdir, "compare.html"))
    print("Done. open", grid, "or", os.path.join(outdir,"compare.html"))

if __name__ == "__main__":
    main()
