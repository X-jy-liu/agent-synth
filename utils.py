from skimage.metrics import structural_similarity as ssim
import numpy as np

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
