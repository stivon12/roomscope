"""Damage candidates per image, run as a subprocess like seg_worker (torch never shares a process with
pycolmap):

    python -m roomscope.damage.worker <job.json> <out.npz>

job.json: {"images": [path, ...], "rot90": [k, ...], "classes": {class: [phrase, ...]},
           "describe": {class: sentence}, "negatives": [sentence, ...], "class_threshold": 0.7,
           "box_threshold": 0.2, "text_threshold": 0.15, "work_px": 1024, "max_box_frac": 0.5}
- rot90[i]: counter-clockwise quarter turns that make image i upright (phones store sensor orientation);
  boxes and masks are returned in the stored orientation.
- Three ungated Apache-2.0 models: Grounding DINO proposes boxes for the class phrases, SigLIP 2 classifies each
  box (crop widened 15 %) against the damage descriptions AND negative descriptions (clean wall, outlet, lamp,
  shadow, frame, ...), and SAM 2.1 turns kept boxes into masks. Grounding DINO alone does not separate the classes:
  on real photos it put boxes for every phrase on almost every image, clean rooms included
  (bench/damage_photos.py). A box is kept when the classifier's best label is a damage class with probability
  >= class_threshold; that label is the class.
- Boxes covering more than max_box_frac of the image are dropped: the phrases also fire on whole walls.

out.npz per image i (at the reduced working size, stored orientation):
  det_<i>: (n, 6) float32 = class index, classifier probability, x0, y0, x1, y1;  mask_<i>: (n, h, w) bool;  hw_<i>: (h, w)
plus `classes` (names in index order) and `model` (which detector ran).
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

GDINO_ID = "IDEA-Research/grounding-dino-base"
SIGLIP_ID = "google/siglip2-base-patch16-384"
SAM2_ID = "facebook/sam2.1-hiera-small"


def run(job: dict) -> dict:
    import cv2
    import torch
    from PIL import Image

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    classes: dict[str, list[str]] = job["classes"]
    names = list(classes)
    work_px = job.get("work_px", 1024)
    max_frac = job.get("max_box_frac", 0.5)
    out = {"classes": np.array(names)}
    from transformers import (AutoModel, AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model,
                              Sam2Processor)
    gp = AutoProcessor.from_pretrained(GDINO_ID)
    gm = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(dev).eval()
    sp = Sam2Processor.from_pretrained(SAM2_ID)
    sm = Sam2Model.from_pretrained(SAM2_ID).to(dev).eval()
    cp = AutoProcessor.from_pretrained(SIGLIP_ID)
    cm = AutoModel.from_pretrained(SIGLIP_ID).to(dev).eval()
    texts = [job["describe"][c] for c in names] + list(job["negatives"])
    with torch.no_grad():
        te = cm.get_text_features(**cp(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(dev))
        te = getattr(te, "pooler_output", te)
        te = te / te.norm(dim=-1, keepdim=True)

    def classify(crop):
        with torch.no_grad():
            ie = cm.get_image_features(**cp(images=crop, return_tensors="pt").to(dev))
            ie = getattr(ie, "pooler_output", ie)
            ie = ie / ie.norm(dim=-1, keepdim=True)
            return torch.softmax((ie @ te.T) * cm.logit_scale.exp() + cm.logit_bias, -1)[0].cpu().numpy()

    out["model"] = np.array(f"{GDINO_ID} + {SIGLIP_ID} + {SAM2_ID}")
    text = " ".join(f"{p}." for ph in classes.values() for p in ph)
    t0 = time.time()
    for i, (path, k) in enumerate(zip(job["images"], job["rot90"])):
        img = np.asarray(Image.open(path).convert("RGB"))
        f = work_px / max(img.shape[:2])
        if f < 1:
            img = cv2.resize(img, (round(img.shape[1] * f), round(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
        up = np.ascontiguousarray(np.rot90(img, k))
        H, W = up.shape[:2]
        with torch.no_grad():
            gi = gp(images=Image.fromarray(up), text=text, return_tensors="pt").to(dev)
            go = gm(**gi)
        r = gp.post_process_grounded_object_detection(go, gi["input_ids"], threshold=job.get("box_threshold", 0.2),
                                                      text_threshold=job.get("text_threshold", 0.15),
                                                      target_sizes=[(H, W)])[0]
        dets = []
        pil = Image.fromarray(up)
        for x0, y0, x1, y1 in r["boxes"].cpu().numpy():
            w, h = x1 - x0, y1 - y0
            if w * h > max_frac * W * H or w < 8 or h < 8:
                continue
            pr = classify(pil.crop((max(0, x0 - .15 * w), max(0, y0 - .15 * h), min(W, x1 + .15 * w), min(H, y1 + .15 * h))))
            ci = int(np.argmax(pr))
            if ci < len(names) and pr[ci] >= job.get("class_threshold", 0.7):
                dets.append([ci, float(pr[ci]), x0, y0, x1, y1])
        masks = np.zeros((0, H, W), bool)
        if dets:
            boxes = [[[float(v) for v in d[2:]] for d in dets]]
            with torch.no_grad():
                si = sp(images=Image.fromarray(up), input_boxes=boxes, return_tensors="pt").to(dev)
                so = sm(**si, multimask_output=False)
            masks = sp.post_process_masks(so.pred_masks.cpu(), si["original_sizes"].cpu())[0][:, 0].numpy() > 0
        # back to the stored orientation
        masks = np.ascontiguousarray(np.rot90(masks, -k, axes=(1, 2)))
        dets = np.array(dets, np.float32).reshape(-1, 6)
        if k % 4:                                # boxes in the stored orientation, from the turned-back masks
            for j in range(len(dets)):
                ys, xs = np.nonzero(masks[j])
                if len(xs):
                    dets[j, 2:] = [xs.min(), ys.min(), xs.max(), ys.max()]
        out[f"det_{i}"] = dets
        out[f"mask_{i}"] = masks
        out[f"hw_{i}"] = np.array(masks.shape[1:] if len(masks) else np.rot90(up, -k).shape[:2])
    print(f"[damage] {len(job['images'])} images in {time.time() - t0:.1f} s on {dev}", file=sys.stderr)
    return out


if __name__ == "__main__":
    job = json.loads(open(sys.argv[1]).read())
    np.savez_compressed(sys.argv[2], **run(job))
