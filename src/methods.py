import matplotlib.pyplot as plt
from utils import _norm_for
import numpy as np
import cv2

def sift_kps_images(rgb_triplet):
    sift = cv2.SIFT_create()
    keypoints_list, descriptors_list, visual_list = [], [], []
    for img_rgb in rgb_triplet:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        visual_bgr = cv2.drawKeypoints(img_bgr, keypoints, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        visual_rgb = cv2.cvtColor(visual_bgr, cv2.COLOR_BGR2RGB)
        visual_list.append(visual_rgb)
    return keypoints_list, descriptors_list, visual_list

def anms(keypoints, descriptors, num_points):
    if len(keypoints) <= num_points:
        return keypoints, descriptors
    
    # inicializo los radios de supresión
    radii = [float('inf')] * len(keypoints)

    # calculo los radios de supresión
    for i in range(len(keypoints)):
        for j in range(len(keypoints)):
            if keypoints[j].response > keypoints[i].response:
                dx = keypoints[i].pt[0] - keypoints[j].pt[0]
                dy = keypoints[i].pt[1] - keypoints[j].pt[1]
                dist = dx*dx + dy*dy
                if dist < radii[i]:
                    radii[i] = dist

    # asocio cada radio con su keypoint y ordeno x radio
    keypoints_radii = list(zip(keypoints, radii))
    keypoints_radii.sort(key=lambda x: x[1], reverse=True)

    # selecciono los primeros num_points keypoints y sus descriptores
    selected_keypoints = [kp for kp, r in keypoints_radii[:num_points]]
    selected_indices = [keypoints.index(kp) for kp in selected_keypoints]
    selected_descriptors = descriptors[selected_indices]
    selected_radii = [r for kp, r in keypoints_radii[:num_points]]

    return selected_keypoints, selected_descriptors, selected_radii

def apply_anms_triplet(kps_list, desc_list, N=1000):
    kps_anms, desc_anms, radii = [], [], []
    for kps, desc in zip(kps_list, desc_list):
        k_sel, d_sel, r_sel = anms(kps, desc, N)
        kps_anms.append(k_sel)
        desc_anms.append(d_sel)
        radii.append(r_sel)
    return kps_anms, desc_anms, radii

# ratio de lowe
def match_lowe(keypoints1, descriptors1, keypoints2, descriptors2, ratio=0.65):
    if descriptors1 is None or descriptors2 is None or len(descriptors1)==0 or len(descriptors2)==0:
        return [], np.empty((0,2), np.float32), np.empty((0,2), np.float32)
    matcher = cv2.BFMatcher(_norm_for(descriptors1), crossCheck=False)
    knn = matcher.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for pair in knn:
        if len(pair)<2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    good.sort(key=lambda x: x.distance)
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good], np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good], np.float32)
    return good, pts1, pts2

# crosscheck
def match_crosscheck(keypoints1, descriptors1, keypoints2, descriptors2):
    if descriptors1 is None or descriptors2 is None or len(descriptors1)==0 or len(descriptors2)==0:
        return [], np.empty((0,2), np.float32), np.empty((0,2), np.float32)
    matcher = cv2.BFMatcher(_norm_for(descriptors2), crossCheck=True)
    matches = matcher.match(descriptors2, descriptors2)
    matches.sort(key=lambda x: x.distance)
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches], np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches], np.float32)
    return matches, pts1, pts2

# lowe + crosscheck
def match_lowe_cross(keypoints1, descriptors1, keypoints2, descriptors2, ratio=0.65):
    if descriptors1 is None or descriptors2 is None or len(descriptors1)==0 or len(descriptors2)==0:
        return [], np.empty((0,2), np.float32), np.empty((0,2), np.float32)
    
    norm = _norm_for(descriptors1)
    bf12 = cv2.BFMatcher(norm, crossCheck=False)
    knn12 = bf12.knnMatch(descriptors1, descriptors2, k=2)
    cand = []
    for pair in knn12:
        if len(pair)<2: 
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            cand.append(m)

    bf21 = cv2.BFMatcher(norm, crossCheck=False)
    knn21 = bf21.knnMatch(descriptors2, descriptors1, k=1)
    best21 = {m.queryIdx: m.trainIdx for [m] in knn21}
    good = [m for m in cand if best21.get(m.trainIdx, -1) == m.queryIdx]
    good.sort(key=lambda x: x.distance)
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in good], np.float32)
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in good], np.float32)
    return good, pts1, pts2
 
def compute_homography(src_pts, dst_pts):
    A = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp])
        A.append([ 0,  0,  0,-x,-y,-1, x*yp, y*yp, yp])

    A = np.asarray(A, dtype=float)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3,3)
    return H / H[2,2]

def warp_to_center_and_overlay(moving_rgb, center_rgb, src_pts, dst_pts, title='', alpha=0.5):
    H = compute_homography(np.asarray(src_pts, float), np.asarray(dst_pts, float))
    h, w = center_rgb.shape[:2]
    warped = cv2.warpPerspective(moving_rgb, H, (w, h)) # esto hace canal por canal
    overlay = (alpha*warped + (1-alpha)*center_rgb).astype(np.uint8)

    plt.figure(figsize=(8,6))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    return H, warped, overlay

def ransac_homography(pts1, pts2, thresh=5.0, max_iters=1000, confidence=0.99):
    best_H = None
    best_inliers = []
    n_points = pts1.shape[0]
    best_inlier_count = 0

    for _ in range(max_iters):
        # seleccionamos 4 puntos aleatorios
        indices = np.random.choice(n_points, 4, replace=False)
        src_sample = pts1[indices]
        dst_sample = pts2[indices]
        
        # calculamos la homografía
        H = compute_homography(src_sample, dst_sample)
        
        # proyectamos todos los puntos
        pts1_homog = np.hstack([pts1, np.ones((n_points, 1))])
        projected_pts2_homog = (H @ pts1_homog.T).T
        projected_pts2 = projected_pts2_homog[:, :2] / projected_pts2_homog[:, 2:3]
        
        # ccalculamos los errores
        errors = np.linalg.norm(projected_pts2 - pts2, axis=1)
        
        # determinamos inliers
        inliers = np.where(errors < thresh)[0]
        inlier_count = len(inliers)
        
        # actualizamos
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inliers
            best_H = H
            
            inlier_ratio = inlier_count / n_points
            p_no_outliers = 1 - inlier_ratio**4
            p_no_outliers = max(p_no_outliers, 1e-8)  # esto para evitar log(0)
            max_iters = min(max_iters, int(np.log(1 - confidence) / np.log(p_no_outliers)))
    
    # si hay suficientes inliers, volvemos a calcular la homografía
    if len(best_inliers) >= 4:
        best_H = compute_homography(pts1[best_inliers], pts2[best_inliers])
    
    return best_H, best_inliers