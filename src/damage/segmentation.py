"""
segmentation.py — Step 2
Per-frame damage segmentation.

Backend priority:
  1. SAM + classifier  (if segment_anything is installed)
  2. SegFormer         (if transformers is installed)
  3. HSV colour-threshold fallback  (always available)

The fallback is deterministic and good enough for a working pipeline.
Swap in the model backends by installing the packages — no code changes needed.

Each backend returns the same format:
    list of dicts  { 'class': str, 'mask': (H,W) bool, 'confidence': float }
"""

import numpy as np
import cv2

DAMAGE_CLASSES = ["water_stain", "crack", "mould", "peeling_paint", "efflorescence"]


# ── Backend: SAM + classifier ─────────────────────────────────────────────────

def _segment_with_sam(bgr: np.ndarray) -> list[dict]:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    import torch
    # Requires a SAM checkpoint; checkpoint path should be set in env SAM_CHECKPOINT
    import os
    ckpt = os.getenv("SAM_CHECKPOINT", "models/sam_vit_h.pth")
    sam  = sam_model_registry["vit_h"](checkpoint=ckpt)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    gen  = SamAutomaticMaskGenerator(sam)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    masks = gen.generate(rgb)

    results = []
    for m in masks:
        mask = m["segmentation"]
        # Crop the masked region and classify by dominant colour hue
        region = bgr[mask]
        cls, conf = _classify_region_by_colour(region)
        if cls:
            results.append({"class": cls, "mask": mask, "confidence": conf})
    return results


# ── Backend: SegFormer ────────────────────────────────────────────────────────

def _segment_with_segformer(bgr: np.ndarray) -> list[dict]:
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    import torch, os
    model_name = os.getenv("SEGFORMER_MODEL", "nvidia/segformer-b0-finetuned-ade-512-512")
    processor  = SegformerImageProcessor.from_pretrained(model_name)
    model      = SegformerForSemanticSegmentation.from_pretrained(model_name)
    rgb        = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inputs     = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits   # (1, num_labels, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits, size=rgb.shape[:2], mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1)[0].numpy()
    # Map generic ade20k labels to damage classes as best-effort
    # (a fine-tuned model on a damage dataset would be more accurate)
    results = []
    for label_id in np.unique(pred):
        mask = (pred == label_id)
        region = bgr[mask]
        cls, conf = _classify_region_by_colour(region)
        if cls:
            results.append({"class": cls, "mask": mask, "confidence": conf})
    return results


# ── Colour-threshold fallback ─────────────────────────────────────────────────

def _classify_region_by_colour(region_bgr: np.ndarray) -> tuple[str | None, float]:
    """
    Heuristic damage class from mean HSV of a region.
    Returns (class_name, confidence) or (None, 0) if no damage class fits.
    """
    if len(region_bgr) == 0:
        return None, 0.0

    hsv  = cv2.cvtColor(region_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h    = float(hsv[:, 0].mean())    # 0–180 in OpenCV
    s    = float(hsv[:, 1].mean())    # 0–255
    v    = float(hsv[:, 2].mean())

    # Water stain: low saturation, mid-dark value, brownish/yellowish hue
    if 10 <= h <= 30 and s > 40 and v < 180:
        return "water_stain", 0.65
    # Mould: greenish or dark low-saturation patches
    if (40 <= h <= 80 and s > 50) or (s < 30 and v < 80):
        return "mould", 0.60
    # Crack: very dark, low saturation, thin regions — handled by morphology below
    if s < 25 and v < 60:
        return "crack", 0.55
    # Peeling paint: high brightness, patchy texture — hard to detect by colour alone
    if s < 40 and v > 200:
        return "peeling_paint", 0.50
    return None, 0.0


def _fallback_colour_threshold(bgr: np.ndarray, process_width: int = 320) -> list[dict]:
    """
    Detect damage regions purely from HSV colour thresholds + connected components.
    Works without any ML model.

    Runs on a downsampled copy for speed; masks are scaled back to original resolution.
    process_width controls the working resolution (lower = faster, coarser masks).
    """
    orig_h, orig_w = bgr.shape[:2]
    scale = process_width / orig_w
    small_w = process_width
    small_h = int(orig_h * scale)

    small = cv2.resize(bgr, (small_w, small_h), interpolation=cv2.INTER_AREA)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    results = []

    stain_mask = cv2.inRange(hsv, (10, 40, 60),  (30, 200, 200))
    mould_mask = cv2.inRange(hsv, (40, 50, 20),  (80, 255, 120))
    crack_mask = cv2.inRange(hsv, (0,   0,  0),  (180, 40,  60))

    # Scale the minimum pixel area threshold to the downsampled resolution
    min_area_small = int(500 * scale * scale)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for raw_mask, cls in [(stain_mask, "water_stain"),
                          (mould_mask, "mould"),
                          (crack_mask, "crack")]:
        cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  kernel)
        cleaned = cv2.morphologyEx(cleaned,  cv2.MORPH_CLOSE, kernel)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
        for lbl in range(1, n_labels):
            if int(stats[lbl, cv2.CC_STAT_AREA]) < max(min_area_small, 10):
                continue
            small_mask = (labels == lbl).astype(np.uint8)
            # Scale mask back to original resolution
            full_mask = cv2.resize(small_mask, (orig_w, orig_h),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            results.append({"class": cls, "mask": full_mask, "confidence": 0.55})

    return results


# ── Public API ────────────────────────────────────────────────────────────────

def detect_damage(bgr: np.ndarray, process_width: int = 320) -> list[dict]:
    """
    Run damage segmentation on a single BGR frame.
    Tries SAM → SegFormer → colour-threshold fallback in order.
    """
    try:
        return _segment_with_sam(bgr)
    except Exception:
        pass
    try:
        return _segment_with_segformer(bgr)
    except Exception:
        pass
    return _fallback_colour_threshold(bgr, process_width=process_width)
