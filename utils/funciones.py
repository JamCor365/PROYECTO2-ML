import numpy as np
from collections import deque
import cv2
import mahotas
# --- K-means++ init ---
def kmeans_plus_plus_init(X, k, random_state=None):
    np.random.seed(random_state)
    n_samples, _ = X.shape
    centers = []
    centers.append(X[np.random.randint(n_samples)])
    for _ in range(1, k):
        dist_sq = np.min(np.linalg.norm(X[:, None] - np.array(centers)[None, :], axis=2)**2, axis=1)
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = np.cumsum(probs)
        r = np.random.rand()
        idx = np.searchsorted(cumulative_probs, r)
        centers.append(X[idx])
    return np.array(centers)

# --- K-means completo ---
def kmeans(X, k, max_iter=100, tol=1e-4, random_state=None):
    centers = kmeans_plus_plus_init(X, k, random_state)
    labels = np.zeros(X.shape[0], dtype=int)

    for it in range(max_iter):
        dists = np.linalg.norm(X[:, None] - centers[None, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = np.array([X[new_labels == i].mean(axis=0) if np.any(new_labels == i) else centers[i] for i in range(k)])

        if np.all(labels == new_labels) or np.linalg.norm(new_centers - centers) < tol:
            break

        labels = new_labels
        centers = new_centers

    return labels, centers


# --- DBSCAN manual ---
def dbscan(X, eps, min_samples):
    n_samples = X.shape[0]
    labels = np.full(n_samples, -1)  # -1 = ruido
    cluster_id = 0

    def region_query(idx):
        dists = np.linalg.norm(X - X[idx], axis=1)
        return np.where(dists <= eps)[0]

    def expand_cluster(idx, neighbors):
        labels[idx] = cluster_id
        queue = deque(neighbors)
        while queue:
            current = queue.popleft()
            if labels[current] == -1:
                labels[current] = cluster_id
            if labels[current] != -1:
                continue
            labels[current] = cluster_id
            current_neighbors = region_query(current)
            if len(current_neighbors) >= min_samples:
                queue.extend(current_neighbors)

    for i in range(n_samples):
        if labels[i] != -1:
            continue
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            labels[i] = -1  # ruido temporal
        else:
            expand_cluster(i, neighbors)
            cluster_id += 1

    return labels
def extract_features(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.resize(img, (182, 268))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # RGB + HSV
    hist_rgb, hist_hsv = [], []
    for i in range(3):
        hist_rgb.extend(cv2.calcHist([img_rgb], [i], None, [64], [0, 256]).flatten())
        hist_hsv.extend(cv2.calcHist([img_hsv], [i], None, [64], [0, 256]).flatten())

    # HOG
    hog_img = cv2.resize(img_gray, (64, 128))
    hog_features = cv2.HOGDescriptor().compute(hog_img).flatten()

    # SIFT
    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(img_gray, None)
    if descriptors is not None:
        sift_features = descriptors[:10].flatten()
        if sift_features.size < 1280:
            sift_features = np.pad(sift_features, (0, 1280 - sift_features.size))
    else:
        sift_features = np.zeros(1280)

    # Zernike
    side = min(img_gray.shape)
    img_square = img_gray[:side, :side]
    img_bin = cv2.threshold(img_square, 128, 255, cv2.THRESH_BINARY)[1]
    zernike_features = mahotas.features.zernike_moments(img_bin, radius=side//2, degree=8)

    return np.concatenate([hist_rgb, hist_hsv, hog_features, sift_features, zernike_features])