import matplotlib.pyplot as plt
from utils import _u8
import numpy as np
import cv2


def show_triplets_2x3(row1, row2, row1_label='Udesa', row2_label='Cuadro', titles=('Left Image','Center Image','Right Image'), figsize=(14,10), wspace=0.08, hspace=0.08):
    fig, ax = plt.subplots(2, 3, figsize=figsize, gridspec_kw={'wspace':wspace, 'hspace':hspace})
    for j in range(3):
        ax[0, j].imshow(row1[j])
        ax[0, j].set_title(f'{row1_label} {titles[j]}')
        ax[0, j].axis('off')
        ax[1, j].imshow(row2[j])
        ax[1, j].set_title(f'{row2_label} {titles[j]}')
        ax[1, j].axis('off')
    plt.show()

def plot_triplet_keypoints(image_triplet, kps_orig_triplet, kps_anms_triplet, title_prefix=''):
    fig, ax = plt.subplots(2, 3, figsize=(14,10), gridspec_kw={'wspace':0.08, 'hspace':0.08})
    names = ['Left', 'Center', 'Right']

    # graficio los originales arriba
    for j in range(3):
        ax[0, j].imshow(image_triplet[j], cmap='gray')
        ax[0, j].axis('off')
        ax[0, j].scatter([kp.pt[0] for kp in kps_orig_triplet[j]], [kp.pt[1] for kp in kps_orig_triplet[j]], s=8, c='g')
        ax[0, j].set_title(f'{title_prefix} {names[j]} - Original')

    # grafico despues de anms abajo
    for j in range(3):
        ax[1, j].imshow(image_triplet[j], cmap='gray')
        ax[1, j].axis('off')
        ax[1, j].scatter([kp.pt[0] for kp in kps_anms_triplet[j]], [kp.pt[1] for kp in kps_anms_triplet[j]], s=8, c='g')
        ax[1, j].set_title(f'{title_prefix} {names[j]} - ANMS')

    plt.tight_layout()
    plt.show()

def show_matches_rgb(img1_rgb, kps1, img2_rgb, kps2, matches, max_show=60, title=''):
    img1_u8 = _u8(img1_rgb)
    img2_u8 = _u8(img2_rgb)
    draw = cv2.drawMatches(cv2.cvtColor(img1_u8, cv2.COLOR_RGB2BGR), kps1, cv2.cvtColor(img2_u8, cv2.COLOR_RGB2BGR), kps2, matches[:max_show], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def draw_matches_thick_rgb(img1_rgb, keypoints1, img2_rgb, keypoints2, matches, max_show=60, thickness=3, radius=6):
    img1 = (img1_rgb*255).astype(np.uint8) if img1_rgb.dtype!=np.uint8 else img1_rgb
    img2 = (img2_rgb*255).astype(np.uint8) if img2_rgb.dtype!=np.uint8 else img2_rgb

    im1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    im2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = im1
    canvas[:h2, w1:w1+w2] = im2
    rng = np.random.default_rng(0)

    for m in matches[:max_show]:
        p1 = tuple(int(v) for v in keypoints1[m.queryIdx].pt)
        p2 = tuple(int(v) for v in keypoints2[m.trainIdx].pt)
        p2_off = (p2[0] + w1, p2[1])
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.circle(canvas, p1, radius, color, -1)
        cv2.circle(canvas, p2_off, radius, color, -1)
        cv2.line(canvas, p1, p2_off, color, thickness, lineType=cv2.LINE_AA)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12,6))
    plt.imshow(canvas_rgb)
    plt.axis('off')
    plt.show()

def plot_overlays_pair_rgb(overlay_left_rgb, overlay_right_rgb, title_left, title_right, figsize=(16,8)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(overlay_left_rgb)
    ax[0].set_title(title_left)
    ax[0].axis('off')
    ax[1].imshow(overlay_right_rgb)
    ax[1].set_title(title_right)
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

def graph_images(images, titles=['Árboles Left Image', 'Árboles Center Image', 'Árboles Right Image'], figsize=(15,5)):
    n = len(images)
    fig, ax = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        ax = [ax]
    for i in range(n):
        ax[i].imshow(images[i])
        ax[i].set_title(titles[i], fontsize=16)
        ax[i].axis('off')
    plt.tight_layout()
    plt.show()

