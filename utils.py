import numpy as np
from PIL import ImageChops

def compute_iou(img1, img2, threshold=128):
    """
    Computes pixel-level IoU between two grayscale PIL images.
    Assumes images are binarized by threshold.
    """
    img1 = img1.convert("L")
    img2 = img2.convert("L")
    arr1 = np.array(img1) > threshold
    arr2 = np.array(img2) > threshold

    intersection = np.logical_and(arr1, arr2).sum()
    union = np.logical_or(arr1, arr2).sum()
    return intersection / union if union > 0 else 0.0

def crop_difference_region(img1, img2, padding=10):
    diff = ImageChops.difference(img1, img2).convert("L")
    bbox = diff.getbbox()
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    x0 = max(x0 - padding, 0)
    y0 = max(y0 - padding, 0)
    x1 = min(x1 + padding, img1.width)
    y1 = min(y1 + padding, img1.height)
    return img1.crop((x0, y0, x1, y1))

