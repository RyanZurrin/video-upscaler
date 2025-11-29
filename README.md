# Video Upscaling and Enhancement Pipeline

A complete GPU-optimized pipeline for video super-resolution, denoising, frame interpolation, and model comparison.  
Designed for extremely low-bitrate sources (240p–360p), anime or live-action, and tuned for RTX 2060 performance.

This repository includes:

- A scriptable pipeline for batch video enhancement
- Automatic model switching (RealESRGAN, SwinIR, BSRGAN)
- Optional Stable Video Diffusion enhancement
- Side-by-side comparison grid generator
- Synchronized comparison video player
- Modular model loading structure

---

## 1. Folder Structure

Your project folder should look like this:

.
├── models
│ ├── bsrgan
│ │ └── BSRGAN.pth
│ ├── realesrgan
│ │ └── realesr-general-x4v3.pth
│ └── swinir
│ └── 003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth
│
├── input_videos
│ └── your_source_files.mp4
│
├── outputs
│ ├── enhanced
│ ├── comparison_grids
│ └── side_by_side
│
├── pipelines
│ ├── upscale_bsrgan.py
│ ├── upscale_swinir.py
│ ├── upscale_realesrgan.py
│ ├── upscale_all.py
│ ├── generate_comparison_grid.py
│ └── compare_player.py
│
└── README.md

---

## 2. Installation

### Install PyTorch (CUDA support):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### Install required Python libraries:

pip install pillow opencv-python tqdm numpy einops ffmpeg-python

### Optional: Diffusion support

pip install diffusers transformers accelerate

You also need FFmpeg:
Windows builds: https://www.gyan.dev/ffmpeg/builds/

---

## 3. Required Models

### RealESRGAN (General x4)

Download from:
https://github.com/xinntao/Real-ESRGAN/releases

Place here:
models/realesrgan/realesr-general-x4v3.pth

---

### SwinIR x4 GAN

Download from:
https://github.com/JingyunLiang/SwinIR/releases

Place here:
models/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth

---

### BSRGAN x4

Download from OpenModelDB.

Place here:
models/bsrgan/BSRGAN.pth

---

## 4. Using the Pipeline

### A. Run all models on a single video

This outputs:

- realesrgan_output.mp4
- swinir_output.mp4
- bsrgan_output.mp4

Command:
python pipelines/upscale_all.py --input input_videos/example.mp4

---

### B. Generate a comparison grid (PNG)

Command:
python pipelines/generate_comparison_grid.py --input input_videos/example.mp4

Output:
outputs/comparison_grids/example_grid.png

---

### C. Launch side-by-side video comparison player

Command:
python pipelines/compare_player.py --videos outputs/enhanced/\*.mp4

Controls:
Space = pause/resume  
Left/Right arrows = seek  
S = save screenshot

---

## 5. Best Models for Low-Bitrate Sources

### Recommended ranking for 240p–360p noisy material:

1. BSRGAN  
   Best realism and detail without excessive artifacts.

2. SwinIR x4 GAN  
   Cleanest results, least hallucinations, soft but natural.

3. RealESRGAN general-x4v3  
   Very sharp but can introduce fake facial features.

### For extremely compressed videos:

Denoise → BSRGAN → SwinIR → (optional) diffusion enhancement

---

## 6. RTX 2060 Performance Tuning

Use these flags for RealESRGAN:
--half --tile 128 --tile_pad 16

General tips:

- Always use float16 on CUDA
- Keep batch size equal to 1
- Use tile sizes between 128 and 192 to avoid VRAM overflow
- Prefer SwinIR for large frames (better memory efficiency)

---

## 7. Roadmap

[ ] RIFE frame interpolation integration  
[ ] Chronos diffusion frame interpolation  
[ ] Add x2, x3, x6, x8 options  
[ ] Web UI  
[ ] VSGAN anime mode

---

## 8. License

MIT License.  
All neural network models retain their original author licenses.

---

## 9. Credits

RealESRGAN by Xintao  
SwinIR by Jingyun Liang  
BSRGAN by C. Zeng  
Stable Video Diffusion by Stability AI

---
