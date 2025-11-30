#!/usr/bin/env python3
"""
launcher_gui.py

Minimal PyQt6 GUI to queue and run experiments using run_all.py.
Install PyQt6 in the environment to run.

Usage:
  python pipelines/launcher_gui.py
"""
import sys
import os
import subprocess
from PyQt6 import QtWidgets, QtCore
from common.paths import INPUT_DIR, MODELS_DIR, OUTPUT_DIR

class Worker(QtCore.QThread):
    append = QtCore.pyqtSignal(str)
    finished_signal = QtCore.pyqtSignal(int)

    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd

    def run(self):
        self.append.emit(f"Running: {self.cmd}\n")
        p = subprocess.Popen(self.cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True)
        for line in p.stdout:
            self.append.emit(line)
        p.wait()
        self.finished_signal.emit(p.returncode)

class Launcher(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Upscale Launcher")
        self.resize(900,600)
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.input_file = QtWidgets.QLineEdit()
        self.input_file.setPlaceholderText("drag input video here or type path")
        form.addRow("Input video:", self.input_file)

        self.realesrgan_bin = QtWidgets.QLineEdit()
        form.addRow("Real-ESRGAN binary:", self.realesrgan_bin)
        self.realesrgan_model = QtWidgets.QLineEdit()
        form.addRow("Real-ESRGAN model folder name:", self.realesrgan_model)
        self.swinir_model = QtWidgets.QLineEdit(os.path.join(MODELS_DIR, "swinir"))
        form.addRow("SwinIR .pth path:", self.swinir_model)
        self.svd_model = QtWidgets.QLineEdit("stabilityai/stable-video-diffusion-img2vid-xt")
        form.addRow("Diffusion model id/path:", self.svd_model)

        layout.addLayout(form)

        h = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run pipeline")
        self.run_btn.clicked.connect(self.start_run)
        h.addWidget(self.run_btn)

        self.open_out = QtWidgets.QPushButton("Open output folder")
        self.open_out.clicked.connect(self.open_outdir)
        h.addWidget(self.open_out)
        layout.addLayout(h)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def append(self, txt):
        self.log.appendPlainText(txt)

    def start_run(self):
        inp = self.input_file.text().strip()
        if not inp:
            self.append("Please specify input video\n"); return
        outdir = os.path.join(OUTPUT_DIR, os.path.splitext(os.path.basename(inp))[0] + "_run")
        cmd = f'python "{os.path.join(os.getcwd(), "pipelines", "run_all.py")}" --input "{inp}" --outdir "{outdir}" --realesrgan_bin "{self.realesrgan_bin.text()}" --realesrgan_model "{self.realesrgan_model.text()}" --swinir_model "{self.swinir_model.text()}" --svd_model "{self.svd_model.text()}"'
        self.worker = Worker(cmd)
        self.worker.append.connect(self.append)
        self.worker.finished_signal.connect(lambda rc: self.append(f"Process finished with code {rc}"))
        self.run_btn.setEnabled(False)
        self.worker.finished_signal.connect(lambda rc: self.run_btn.setEnabled(True))
        self.worker.start()

    def open_outdir(self):
        # open outputs folder
        import subprocess, platform
        if platform.system() == "Windows":
            subprocess.Popen(f'explorer "{OUTPUT_DIR}"')
        else:
            subprocess.Popen(["xdg-open", OUTPUT_DIR])

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Launcher()
    w.show()
    sys.exit(app.exec())
