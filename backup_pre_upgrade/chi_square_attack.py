import cv2
import numpy as np
from scipy.stats import chisquare

def chi_test(image_path):
    img = cv2.imread(image_path, 0)
    pixels = img.flatten()
    hist = np.bincount(pixels, minlength=256)

    expected = np.ones_like(hist) * np.mean(hist)

    chi, p = chisquare(hist, expected)
    return chi, p

# Run on both images
chi_clean, p_clean = chi_test("clean.jpg")
chi_stego, p_stego = chi_test("stego.jpg")

print("---- Chi-Square Comparison ----")
print("Clean  -> Chi:", chi_clean, " p:", p_clean)
print("Stego  -> Chi:", chi_stego, " p:", p_stego)

# Compare instead of absolute decision
if chi_stego > chi_clean:
    print("⚠️ Stego image shows more statistical anomaly")
else:
    print("✅ No strong statistical difference")