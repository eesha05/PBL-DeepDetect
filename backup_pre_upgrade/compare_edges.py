import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load Images
# -----------------------------
clean = cv2.imread("clean.jpg")
stego = cv2.imread("stego.jpg")

# Convert to grayscale
clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)

# -----------------------------
# EDGE DETECTION FUNCTION
# -----------------------------
def detect_edges(image):

    # Sobel
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    sobel = cv2.convertScaleAbs(sobel)

    # Laplacian
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Canny
    canny = cv2.Canny(image, 100, 200)

    return sobel, laplacian, canny


# Apply edge detection
clean_sobel, clean_lap, clean_canny = detect_edges(clean_gray)
stego_sobel, stego_lap, stego_canny = detect_edges(stego_gray)

# -----------------------------
# EDGE DENSITY CALCULATION
# -----------------------------
def edge_density(edge_image):
    return np.sum(edge_image > 0)

print("----- EDGE DENSITY COMPARISON -----")
print("CLEAN IMAGE:")
print("Sobel:", edge_density(clean_sobel))
print("Laplacian:", edge_density(clean_lap))
print("Canny:", edge_density(clean_canny))

print("\nSTEGO IMAGE:")
print("Sobel:", edge_density(stego_sobel))
print("Laplacian:", edge_density(stego_lap))
print("Canny:", edge_density(stego_canny))


# -----------------------------
# DIFFERENCE MAP
# -----------------------------
difference = cv2.absdiff(clean_canny, stego_canny)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(clean_canny, cmap='gray')
plt.title("Clean - Canny")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(stego_canny, cmap='gray')
plt.title("Stego - Canny")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(difference, cmap='gray')
plt.title("Difference Map")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(clean_sobel, cmap='gray')
plt.title("Clean - Sobel")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(stego_sobel, cmap='gray')
plt.title("Stego - Sobel")
plt.axis("off")

plt.tight_layout()
plt.show()
