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
# Difference Attack
# -----------------------------
diff = cv2.absdiff(original, stego)

# Amplify for visibility
diff_amp = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

# Binary map (highlight changed pixels clearly)
_, diff_binary = cv2.threshold(diff, 1, 255, cv2.THRESH_BINARY)

# -----------------------------
# Statistics (IMPORTANT)
# -----------------------------
total_diff = np.sum(diff)
mean_diff = np.mean(diff)
changed_pixels = np.sum(diff_binary > 0)
percentage_changed = (changed_pixels / diff.size) * 100

print("------ Difference Attack Results ------")
print("Total Difference:", total_diff)
print("Mean Difference :", mean_diff)
print("Changed Pixels  :", changed_pixels)
print("Change %        :", percentage_changed)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(original, cmap='gray')
plt.axis('off')

plt.subplot(1,4,2)
plt.title("Stego")
plt.imshow(stego, cmap='gray')
plt.axis('off')

plt.subplot(1,4,3)
plt.title("Difference (Amplified)")
plt.imshow(diff_amp, cmap='hot')
plt.axis('off')

plt.subplot(1,4,4)
plt.title("Changed Pixels")
plt.imshow(diff_binary, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()