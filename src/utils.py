import numpy as np
import cv2

def rgb_to_gray_triplet(rgb_triplet):
    return [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in rgb_triplet]

def _u8(img):
    return (img*255).astype(np.uint8) if img.dtype!=np.uint8 else img

def _norm_for(desc):
    # esto es para elegir la norma correcta segun el descriptor que se use
    return cv2.NORM_L2 if desc.dtype==np.float32 else cv2.NORM_HAMMING

def make_overlay_rgb(base_rgb, over_rgb, alpha=0.5):
    return cv2.addWeighted(base_rgb, 1 - alpha, over_rgb, alpha, 0)

def _corners_of(img):
    h, w = img.shape[:2]
    return np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32)[None, ...]
