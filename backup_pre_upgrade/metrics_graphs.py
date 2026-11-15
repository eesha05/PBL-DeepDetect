import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# -----------------------------
# Load Images
# -----------------------------
original = cv2.imread('clean.jpg', 0)
stego = cv2.imread('stego.jpg', 0)

if original is None or stego is None:
    print("Error loading images")
    exit()

# Resize if needed
if original.shape != stego.shape:
    stego = cv2.resize(stego, (original.shape[1], original.shape[0]))

# -----------------------------
# MSE
# -----------------------------
mse_value = np.mean((original - stego) ** 2)

# -----------------------------
# SSIM
# -----------------------------
ssim_value = ssim(original, stego)

# -----------------------------
# Edge Density
# -----------------------------
def edge_density(img):
    edges = cv2.Canny(img, 100, 200)
    return np.sum(edges > 0) / edges.size

edge_orig = edge_density(original)
edge_stego = edge_density(stego)

# -----------------------------
# Print values
# -----------------------------
print("MSE:", mse_value)
print("SSIM:", ssim_value)
print("Edge Density (Original):", edge_orig)
print("Edge Density (Stego):", edge_stego)

# -----------------------------
# GRAPH 1: MSE & SSIM
# -----------------------------
plt.figure()
labels = ['MSE', 'SSIM']
values = [mse_value, ssim_value]

plt.bar(labels, values)
plt.title("MSE vs SSIM")
plt.ylabel("Value")
plt.show()

# -----------------------------
# GRAPH 2: Edge Density Comparison
# -----------------------------
plt.figure()
labels = ['Original', 'Stego']
values = [edge_orig, edge_stego]

plt.bar(labels, values)
plt.title("Edge Density Comparison")
plt.ylabel("Density")
plt.show()