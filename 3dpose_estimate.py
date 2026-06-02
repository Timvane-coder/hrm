"""
3dpose_estimate.py  —  hmr2.0 drop-in replacement for the original hmr/demo.py

Usage (called by 3dpose_estimate.sh):
    python hmr2.0/3dpose_estimate.py \
        --img_path  <path/to/image.jpg> \
        --json_path <path/to/pose2d.json> \
        --out_dir   <hmr/output/csv>

Output:
    <out_dir>/<image_basename>.csv
    One row per image (frame), columns:
        frame, j0x,j0y,j0z, j1x,j1y,j1z, ... j19x,j19y,j19z   (20 joints × 3 = 60 floats)

    This is exactly the format expected by csv_to_bvh.py / Blender.

Dependencies (all Python 3):
    pip install tensorflow>=2.1 trimesh numpy absl-py opencv-python
    (hmr2.0 repo must be on PYTHONPATH — handled by running from repo root)
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import cv2

# ── hmr2.0 imports ────────────────────────────────────────────────────────────
# Assumes this script lives inside the hmr2.0 clone root, or that
# hmr2.0/src is on PYTHONPATH.
HMR2_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HMR2_ROOT, 'src'))

from main.config import Config
from main.model import Model

# ── constants ─────────────────────────────────────────────────────────────────
# hmr2.0 cocoplus regressor gives 19 keypoints; we need 20 to match the
# original CSV schema.  Joint index mapping (cocoplus → original hmr order):
# The original hmr used the same cocoplus 19-joint set + 1 extra (pelvis/root).
# We reconstruct the 20th joint as the midpoint of left-hip (11) and
# right-hip (12) — identical to how the original demo.py did it.
NUM_JOINTS_OUT = 20

# hmr2.0 model / checkpoint settings — edit these to match your download
MODEL_NAME   = 'base_model'
SETTING      = 'paired(joints)'
JOINT_TYPE   = 'cocoplus'
INIT_TOES    = False


def load_model():
    cfg = Config()
    cfg.JOINT_TYPE  = JOINT_TYPE
    cfg.ENCODER_ONLY = True

    # Point to the log dir where you unpacked the pretrained weights:
    #   logs/paired(joints)/base_model/
    cfg.LOG_DIR = os.path.join(HMR2_ROOT, 'logs', SETTING, MODEL_NAME)

    model = Model()
    return model


def load_image(img_path: str) -> np.ndarray:
    """Load and pre-process image to 224×224 float32 RGB in [0,1]."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img.astype(np.float32) / 255.0


def load_bbox_from_json(json_path: str):
    """
    Read the 2D pose JSON produced by keras_Realtime pipeline.
    Returns (x, y, w, h) bounding box of the detected person,
    or None if the file is missing / has no detections.
    """
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)

    # openpose-style JSON: data['people'][0]['pose_keypoints']
    people = data.get('people', [])
    if not people:
        return None

    kps = np.array(people[0]['pose_keypoints']).reshape(-1, 3)
    xs = kps[kps[:, 2] > 0.1, 0]
    ys = kps[kps[:, 2] > 0.1, 1]
    if len(xs) == 0:
        return None

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    pad = 0.15
    W = x2 - x1
    H = y2 - y1
    return (max(0, x1 - pad * W),
            max(0, y1 - pad * H),
            W * (1 + 2 * pad),
            H * (1 + 2 * pad))


def crop_to_bbox(img_bgr: np.ndarray, bbox) -> np.ndarray:
    """Crop image to bbox and resize to 224×224."""
    x, y, w, h = [int(v) for v in bbox]
    ih, iw = img_bgr.shape[:2]
    x2 = min(iw, x + w)
    y2 = min(ih, y + h)
    crop = img_bgr[max(0, y):y2, max(0, x):x2]
    if crop.size == 0:
        return img_bgr          # fall back to full image
    return crop


def joints_to_row(joints3d: np.ndarray, frame_num: int) -> list:
    """
    Convert (N, 3) joint array to the flat CSV row the Blender script expects:
        [frame, x0, y0, z0, x1, y1, z1, ..., x19, y19, z19]

    joints3d shape from hmr2.0: (19, 3)  (cocoplus)
    We add a synthetic root joint (index 19) = mean of hip joints (11, 12).
    """
    j = joints3d.copy()                      # (19, 3)
    root = ((j[11] + j[12]) / 2.0).reshape(1, 3)
    j = np.vstack([j, root])                 # (20, 3)

    row = [frame_num]
    for jx, jy, jz in j:
        row += [jx, jy, jz]
    return row


def estimate(img_path: str, json_path: str, out_dir: str, frame_num: int, model):
    """Run hmr2.0 on one image and append one CSV row."""
    # ── load & optionally crop to detected person ──────────────────────────
    raw_bgr = cv2.imread(img_path)
    bbox    = load_bbox_from_json(json_path)

    if bbox is not None:
        cropped = crop_to_bbox(raw_bgr, bbox)
    else:
        cropped = raw_bgr

    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0

    # ── inference ──────────────────────────────────────────────────────────
    # model.detect() expects a (224,224,3) float32 array, returns a dict with
    # keys: 'joints3d', 'vertices', 'theta', 'camera', ...
    result = model.detect(img_rgb)

    joints3d = result['joints3d']            # numpy (19, 3)  in camera space (metres)

    # ── write CSV ──────────────────────────────────────────────────────────
    basename  = os.path.splitext(os.path.basename(img_path))[0]
    csv_path  = os.path.join(out_dir, f"{basename}.csv")

    row = joints_to_row(joints3d, frame_num)

    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # header expected by downstream csv_join step
        header = ['frame']
        for i in range(NUM_JOINTS_OUT):
            header += [f'j{i}x', f'j{i}y', f'j{i}z']
        writer.writerow(header)
        writer.writerow(row)

    print(f"  → saved {csv_path}")


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='hmr2.0 3D pose estimation — outputs CSV compatible with csv_to_bvh.py')
    parser.add_argument('--img_path',  required=True,  help='Input image path')
    parser.add_argument('--json_path', required=True,  help='2D pose JSON from keras_Realtime')
    parser.add_argument('--out_dir',   required=True,  help='Output directory for CSV files')
    parser.add_argument('--frame',     type=int, default=0,
                        help='Frame number to write into CSV (default: 0, overridden by shell script)')
    args = parser.parse_args()

    print(f"Loading hmr2.0 model …")
    model = load_model()

    estimate(
        img_path  = args.img_path,
        json_path = args.json_path,
        out_dir   = args.out_dir,
        frame_num = args.frame,
        model     = model,
    )


if __name__ == '__main__':
    main()
  
