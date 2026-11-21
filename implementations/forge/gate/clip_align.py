#!/usr/bin/env python3
"""Geometry check: each 720p clip vs the 2160p master downscaled to 720p (lanczos, code domain).
Sub-pixel shift by upsampled phase correlation on luma; PSNR before/after the best integer-
free shift. A clip made from a different source/scaling than the master shows a shift."""
import os, sys, numpy as np, subprocess, cv2
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO + "/tools"); os.chdir(REPO)
from forge_consistency import decode_yuv
import forge_fquality as Q
C = "research/Temporal-Forge-Player/benchmarks/video_corpus/clips/"; R = "research/Temporal-Forge-Player/benchmarks/video_corpus/references/"
def gt720(scene, n):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", R + f"{scene}_2160p_lossless.mkv", "-vf", f"select=eq(n\\,{n}),scale=1280:720:flags=lanczos",
                          "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(720, 1280).astype(np.float32)
def shift(a, b):
    w = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, w)
    return dx, dy
clips = {"sintel_cave": {"crf10": C + "sintel_cave_1280x720_hq_crf10.mp4", "crf23": C + "sintel_cave_1280x720_medium_crf23.mp4",
                         "crf27": ".scratch/forge/cave-clips/sintel_cave_1280x720_medium_crf27.mp4"},
         "tos_daylight": {"crf10": C + "tos_daylight_1280x720_hq_crf10.mp4", "crf23": C + "tos_daylight_1280x720_medium_crf23.mp4",
                          "crf27": ".scratch/forge/tos-daylight-clips/tos_daylight_1280x720_medium_crf27.mp4"},
         "sintel_rooftop": {"crf10": C + "sintel_rooftop_1280x720_hq_crf10.mp4", "crf23": C + "sintel_rooftop_1280x720_medium_crf23.mp4"},
         "tos_debris": {"crf10": C + "tos_debris_1280x720_hq_crf10.mp4", "crf23": C + "tos_debris_1280x720_medium_crf23.mp4"}}
for scene, arms in clips.items():
    n0 = round(Q._start(arms["crf10"]) * 24)
    for f in (72, 120):
        n = f - n0
        g = gt720(scene, n)
        out = []
        for arm, p in arms.items():
            y = decode_yuv(p, n)[0].numpy().astype(np.float32)
            dx, dy = shift(g, y)
            ps = 10 * np.log10(255 ** 2 / ((y - g) ** 2).mean())
            out.append(f"{arm} shift ({dx:+.3f},{dy:+.3f}) px  PSNR vs master {ps:.2f}")
        print(f"{scene:15s} f{f}: " + " | ".join(out), flush=True)
