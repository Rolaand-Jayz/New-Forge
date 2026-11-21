import os, sys, tempfile, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO + '/tools')
from metrics import decode as mdecode, luma_np
from PIL import Image
TMP = tempfile.mkdtemp(dir=REPO + "/.scratch/forge/gate-confirm")
def m2(a8, b8):
    pa, pb = os.path.join(TMP, "a.png"), os.path.join(TMP, "b.png")
    Image.fromarray(a8).save(pa); Image.fromarray(b8).save(pb)
    y0, y1 = luma_np(mdecode(pa)), luma_np(mdecode(pb))
    gy = np.zeros_like(y0); gx = np.zeros_like(y0)
    gy[1:-1, :] = (y0[2:, :] - y0[:-2, :]) * 0.5; gx[:, 1:-1] = (y0[:, 2:] - y0[:, :-2]) * 0.5
    g = np.hypot(gx, gy); g[:2, :] = g[-2:, :] = 0; g[:, :2] = g[:, -2:] = 0
    mask = g >= np.quantile(g, 0.9)
    return float(np.abs(y1 - y0)[mask].mean())
