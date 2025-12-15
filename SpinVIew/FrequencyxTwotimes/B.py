import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import skew
from scipy.fft import fft2, fftshift

# 1) Load images (ensure natural sorted order matches frequency order)
image_files = sorted(glob.glob("*.bmp"))
images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
labels = [f.split("/")[-1] for f in image_files]  # or replace with actual frequency labels

# 2) Define ROI (center square)
h, w = images[0].shape
roi_size = min(h, w) // 3
x0, y0 = (w - roi_size) // 2, (h - roi_size) // 2

# 3) Feature containers
feat_std = []
feat_roi_mean = []
feat_roi_std = []
feat_entropy = []
feat_skew = []
feat_edge_density = []
feat_fft_total = []
feat_fft_midband = []
feat_ssim = []

# Choose a baseline image (e.g., first one or the quiet/lowest frequency)
baseline = images[0]

# 4) Helper functions
def image_entropy(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256]).flatten()
    p = hist / (np.sum(hist) + 1e-12)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def edge_density(img):
    edges = cv2.Canny(img, 50, 150)
    return np.count_nonzero(edges) / edges.size

def fft_features(img):
    F = fftshift(fft2(img.astype(np.float32)))
    P = np.abs(F)**2
    total = np.sum(P)
    # Midband power: annulus between radii r1 and r2
    yy, xx = np.indices(img.shape)
    cx, cy = w // 2, h // 2
    r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    r1, r2 = min(h,w)*0.05, min(h,w)*0.25
    mid = np.sum(P[(r >= r1) & (r <= r2)])
    return total, mid

# 5) Compute features per image
for img in images:
    roi = img[y0:y0+roi_size, x0:x0+roi_size]
    feat_std.append(np.std(img))
    feat_roi_mean.append(np.mean(roi))
    feat_roi_std.append(np.std(roi))
    feat_entropy.append(image_entropy(img))
    feat_skew.append(skew(img.flatten()))
    feat_edge_density.append(edge_density(img))
    t, m = fft_features(img)
    feat_fft_total.append(t)
    feat_fft_midband.append(m)
    # SSIM vs baseline
    s, _ = ssim(baseline, img, full=True)
    feat_ssim.append(s)

# 6) Normalize features for plotting (optional but helps scale)
def zscore(x):
    x = np.array(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-12)

Z = {
    "Std (global)": zscore(feat_std),
    "ROI mean": zscore(feat_roi_mean),
    "ROI std": zscore(feat_roi_std),
    "Entropy": zscore(feat_entropy),
    "Skewness": zscore(feat_skew),
    "Edge density": zscore(feat_edge_density),
    "FFT total power": zscore(feat_fft_total),
    "FFT midband power": zscore(feat_fft_midband),
    "SSIM vs baseline": zscore(feat_ssim),
}

# 7) Plot all features as grouped bars
# plt.figure(figsize=(14, 8))
# x = np.arange(len(images))
# width = 0.08
# for i, (name, values) in enumerate(Z.items()):
#     plt.bar(x + (i - len(Z)/2)*width, values, width=width, label=name)
# plt.xticks(x, labels, rotation=45)
# plt.ylabel("Z-scored feature value")
# plt.title("Feature comparison across 7 images (different sound frequencies)")
# plt.legend(ncol=3)
# plt.tight_layout()
# plt.show()

# 8) Optional: pairwise difference heatmaps vs baseline
import seaborn as sns
diffs = [cv2.absdiff(img, baseline).astype(np.float32) for img in images]
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
for i, (img, ax) in enumerate(zip(diffs, axs.flat)):
    if i >= len(diffs): break
    sns.heatmap(img, ax=ax, cmap="magma", cbar=False)
    ax.set_title(f"Abs diff vs baseline: {labels[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()