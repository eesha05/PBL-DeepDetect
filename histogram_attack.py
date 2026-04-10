import cv2
import matplotlib.pyplot as plt

# -----------------------------
# Load Images
# -----------------------------
original = cv2.imread('clean.jpg', 0)
stego = cv2.imread('stego.jpg', 0)

if original is None or stego is None:
    print("Error: Check image paths!")
    exit()

# -----------------------------
# Plot Images + Histograms
# -----------------------------
plt.figure(figsize=(12,8))

# Original Image
plt.subplot(2,2,1)
plt.title("Original Image")
plt.imshow(original, cmap='gray')
plt.axis('off')

# Stego Image
plt.subplot(2,2,2)
plt.title("Stego Image")
plt.imshow(stego, cmap='gray')
plt.axis('off')

# Histogram - Original
plt.subplot(2,2,3)
plt.title("Histogram (Original)")
plt.hist(original.ravel(), bins=256)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Histogram - Stego
plt.subplot(2,2,4)
plt.title("Histogram (Stego)")
plt.hist(stego.ravel(), bins=256)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()