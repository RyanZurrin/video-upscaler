#!/usr/bin/env python3
# compare_player.py
import cv2
import argparse
import numpy as np

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--size", nargs=2, type=int, default=[640,360])
    args = p.parse_args()
    vids = [cv2.VideoCapture(f) for f in args.files]
    fps = [v.get(cv2.CAP_PROP_FPS) or 30 for v in vids]
    target_fps = min(fps)
    delay = int(1000.0 / target_fps)
    win = "Compare"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        frames = []
        eof = False
        for v in vids:
            ret, frame = v.read()
            if not ret:
                eof = True
                break
            frame = cv2.resize(frame, tuple(args.size))
            frames.append(frame)
        if eof:
            break
        while len(frames) < 4:
            frames.append(np.zeros_like(frames[0]))
        top = np.hstack((frames[0], frames[1]))
        bottom = np.hstack((frames[2], frames[3]))
        grid = np.vstack((top, bottom))
        cv2.imshow(win, grid)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q') or key == 27:
            break
    for v in vids:
        v.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
