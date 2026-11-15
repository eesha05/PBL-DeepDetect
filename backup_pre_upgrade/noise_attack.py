import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load Images
# -----------------------------
original = cv2.imread('clean.jpg', 0)
stego = cv2.imread('stego.jpg', 0)

if original is None or stego is None:
    print("Error: Check image paths!")
    exit()

# Resize if needed
if original.shape != stego.shape:
    stego = cv2.resize(stego, (original.shape[1], original.shape[0]))

# -----------------------------
# Noise Residual Function (Improved)
# -----------------------------
def get_residual(img):
    blur = cv2.GaussianBlur(img, (3,3), 0)  # smaller kernel = more sensitive
    residual = cv2.absdiff(img, blur)
    return residual

# Compute residuals
residual_orig = get_residual(original)
residual_stego = get_residual(stego)

# -----------------------------
# Amplify for better visibility
# -----------------------------
residual_orig_amp = cv2.normalize(residual_orig, None, 0, 255, cv2.NORM_MINMAX)
residual_stego_amp = cv2.normalize(residual_stego, None, 0, 255, cv2.NORM_MINMAX)

# Difference between residuals (VERY IMPORTANT)
residual_diff = cv2.absdiff(residual_orig, residual_stego)
residual_diff_amp = cv2.normalize(residual_diff, None, 0, 255, cv2.NORM_MINMAX)

# -----------------------------
# Statistics (Mean + STD)
# -----------------------------
print("------ Noise Analysis ------")
print("Original Mean:", np.mean(residual_orig))
print("Stego Mean   :", np.mean(residual_stego))

print("Original STD :", np.std(residual_orig))
print("Stego STD    :", np.std(residual_stego))

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12,8))

# Row 1
plt.subplot(2,3,1)
plt.title("Original")
plt.imshow(original, cmap='gray')
plt.axis('off')

plt.subplot(2,3,2)
plt.title("Stego")
plt.imshow(stego, cmap='gray')
plt.axis('off')

plt.subplot(2,3,3)
plt.title("Residual Difference")
plt.imshow(residual_diff_amp, cmap='hot')
plt.axis('off')

# Row 2
plt.subplot(2,3,4)
plt.title("Original Residual")
plt.imshow(residual_orig_amp, cmap='hot')
plt.axis('off')

plt.subplot(2,3,5)
plt.title("Stego Residual")
plt.imshow(residual_stego_amp, cmap='hot')
plt.axis('off')

plt.tight_layout()
plt.show()