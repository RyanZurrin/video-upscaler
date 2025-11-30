#!/usr/bin/env python3
"""
parallel_runner.py

Run shell job commands in parallel, assigning GPUs (via CUDA_VISIBLE_DEVICES) round-robin.
Safe fallback: if no GPUs or only 1 GPU, jobs run sequentially.

Usage:
  python parallel_runner.py --commands commands.txt --gpus 0,1 --jobs 2
where commands.txt has one shell command per line.
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time

def run_cmd(cmd, gpu_id=None, env_extra=None):
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if env_extra:
        env.update(env_extra)
    start = time()
    p = subprocess.Popen(cmd, shell=True, env=env)
    rc = p.wait()
    elapsed = time() - start
    return rc, elapsed, cmd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commands", required=True)
    p.add_argument("--gpus", default="", help="comma separated GPU ids to use, e.g. 0,1. If empty uses CPU/sequential.")
    p.add_argument("--jobs", type=int, default=1, help="max concurrent jobs")
    args = p.parse_args()

    with open(args.commands, "r", encoding="utf-8") as fh:
        cmds = [line.strip() for line in fh if line.strip()]

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()!=""]
    use_gpus = len(gpu_list) > 0
    max_workers = min(args.jobs, len(cmds))

    results = []
    if use_gpus:
        # map worker idx -> gpu id round robin
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = []
            for i, cmd in enumerate(cmds):
                gpu_id = gpu_list[i % len(gpu_list)]
                futures.append(ex.submit(run_cmd, cmd, gpu_id))
            for fut in as_completed(futures):
                rc, elapsed, cmd = fut.result()
                results.append((rc, elapsed, cmd))
                print(f"[{rc}] {elapsed:.1f}s  {cmd}")
    else:
        # sequential fallback
        for cmd in cmds:
            rc, elapsed, _ = run_cmd(cmd, gpu_id=None)
            results.append((rc, elapsed, cmd))
            print(f"[{rc}] {elapsed:.1f}s  {cmd}")

if __name__ == "__main__":
    main()
