"""Indoor semantic segmentation (EoMT-L, ADE20K-150, MIT licence), run as a subprocess like da3_worker:

    python -m roomscope.frontends.seg_worker <job.json> <out.npz>

job.json: {"images": [path, ...], "rot90": [k, ...], "target": [[W, H], ...]}
- rot90[i]: counter-clockwise quarter turns that make image i upright (phones store frames in sensor
  orientation; the model was trained on upright photos). Labels are turned back afterwards.
- target[i]: size of the label map to return, in the stored (not rotated) orientation; the depth map's size,
  so labels index depth pixels directly.

out.npz per image i: label_<i> uint8 ADE20K class id (0 = wall ... 149), conf_<i> uint8 = 255 x (top class
score / sum of class scores), and `names` (the 150 class names). The model sees each frame at 512 px
(the checkpoint's preprocessing) after the frame is reduced to 640 px, on MPS when available.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

MODEL_ID = "tue-mps/ade20k_semantic_eomt_large_512"
WORK_PX = 640          # long side the frame is reduced to before inference (label maps are depth-sized anyway)


def run(job: dict) -> dict:
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, EomtForUniversalSegmentation

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = EomtForUniversalSegmentation.from_pretrained(MODEL_ID).to(dev).eval()
    names = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    out = {"names": np.array(names)}
    t0 = time.time()
    for i, (path, k, (W, H)) in enumerate(zip(job["images"], job["rot90"], job["target"])):
        img = np.asarray(Image.open(path).convert("RGB"))
        f = WORK_PX / max(img.shape[:2])            # the model sees 512 px; scores at full size cost GBs
        if f < 1:
            img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
        up = np.ascontiguousarray(np.rot90(img, k))
        inputs = proc(images=Image.fromarray(up), return_tensors="pt").to(dev)
        with torch.no_grad():
            o = model(**inputs)
        r = proc.post_process_semantic_segmentation(o, target_sizes=[up.shape[:2]],
                                                    return_segmentation_scores=True)[0]
        s = r.segmentation_scores.float().cpu().numpy()             # (C, h, w)
        lab = s.argmax(0).astype(np.uint8)
        conf = (255 * s.max(0) / np.maximum(s.sum(0), 1e-9)).astype(np.uint8)
        lab, conf = np.rot90(lab, -k), np.rot90(conf, -k)           # back to the stored orientation
        out[f"label_{i}"] = cv2.resize(np.ascontiguousarray(lab), (W, H), interpolation=cv2.INTER_NEAREST)
        out[f"conf_{i}"] = cv2.resize(np.ascontiguousarray(conf), (W, H), interpolation=cv2.INTER_NEAREST)
    print(f"[seg] {len(job['images'])} frames in {time.time() - t0:.1f} s on {dev}", file=sys.stderr)
    return out


if __name__ == "__main__":
    job = json.loads(open(sys.argv[1]).read())
    np.savez_compressed(sys.argv[2], **run(job))
