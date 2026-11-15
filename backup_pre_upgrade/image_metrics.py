
import cv2
import numpy as np
import os

# -----------------------------
# 1. SETTINGS
# -----------------------------
INPUT_IMAGE = "clean.jpg"
OUTPUT_IMAGE = "stego.png"

MESSAGE = "DeepDetect steganography test"
JPEG_QUALITY = 90

# -----------------------------
# 2. LOAD ORIGINAL IMAGE
# -----------------------------
original = cv2.imread(INPUT_IMAGE)

if original is None:
    raise FileNotFoundError(
        f"Cannot find {INPUT_IMAGE}. "
        "Place it in the project folder."
    )

height, width, channels = original.shape
total_pixels = height * width

# We embed one bit per pixel in the blue channel.
message_bytes = MESSAGE.encode("utf-8")
message_bits = np.unpackbits(
    np.frombuffer(message_bytes, dtype=np.uint8)
)

if len(message_bits) > total_pixels:
    raise ValueError(
        "Message is too large for this image."
    )

# -----------------------------
# 3. EMBED MESSAGE USING LSB
# -----------------------------
stego = original.copy()

blue = stego[:, :, 0].flatten()

# Replace each pixel's least significant bit.
blue[:len(message_bits)] = (
    (blue[:len(message_bits)] & 254)
    | message_bits
)

stego[:, :, 0] = blue.reshape(height, width)

cv2.imwrite(OUTPUT_IMAGE, stego)

print("Message embedded successfully.")
print("Stego image saved as:", OUTPUT_IMAGE)

# -----------------------------
# 4. MSE AND PSNR
# -----------------------------
def calculate_mse(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    return np.mean((img1 - img2) ** 2)


def calculate_psnr(mse):
    if mse == 0:
        return float("inf")

    return 10 * np.log10((255 ** 2) / mse)


mse = calculate_mse(original, stego)
psnr = calculate_psnr(mse)

print("\n--- IMAGE QUALITY ---")
print(f"MSE: {mse:.6f}")
print(f"PSNR: {psnr:.4f} dB")

# -----------------------------
# 5. EMBEDDING CAPACITY
# -----------------------------
embedded_bits = len(message_bits)

# Maximum capacity of this simple method:
# one bit per pixel in the blue channel.
maximum_capacity_bits = total_pixels

capacity_bpp = embedded_bits / total_pixels

capacity_percentage = (
    embedded_bits / maximum_capacity_bits
) * 100

print("\n--- EMBEDDING CAPACITY ---")
print("Image dimensions:", width, "x", height)
print("Total pixels:", total_pixels)
print("Embedded bits:", embedded_bits)
print("Embedded bytes:", len(message_bytes))
print(f"Payload rate: {capacity_bpp:.6f} bits/pixel")
print(f"Capacity used: {capacity_percentage:.4f}%")
print(
    "Maximum capacity:",
    maximum_capacity_bits,
    "bits"
)

# -----------------------------
# 6. EXTRACT EMBEDDED BITS
# -----------------------------
def extract_bits(image, number_of_bits):
    blue_channel = image[:, :, 0].flatten()
    return blue_channel[:number_of_bits] & 1


def calculate_ber(expected_bits, received_bits):
    errors = np.count_nonzero(
        expected_bits != received_bits
    )

    ber = errors / len(expected_bits)

    return errors, ber


extracted_bits = extract_bits(stego, embedded_bits)
errors, ber = calculate_ber(message_bits, extracted_bits)

print("\n--- MESSAGE RECOVERY ---")
print("Incorrect bits:", errors)
print(f"BER: {ber:.6f}")

# -----------------------------
# 7. ROBUSTNESS TESTS
# -----------------------------
def test_robustness(name, transformed_image):
    received_bits = extract_bits(
        transformed_image,
        embedded_bits
    )

    errors, ber = calculate_ber(
        message_bits,
        received_bits
    )

    print(f"\n{name}")
    print(f"BER: {ber:.6f}")
    print(f"Bit error percentage: {ber * 100:.2f}%")
    print(f"Incorrect bits: {errors}/{embedded_bits}")


print("\n--- ROBUSTNESS TESTING ---")

# Test A: JPEG recompression
jpeg_path = "test_jpeg.jpg"

cv2.imwrite(
    jpeg_path,
    stego,
    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
)

jpeg_image = cv2.imread(jpeg_path)
test_robustness("JPEG recompression", jpeg_image)

# Test B: Resize down and then back to original size
small = cv2.resize(
    stego,
    (max(1, width // 2), max(1, height // 2))
)

resized_back = cv2.resize(
    small,
    (width, height),
    interpolation=cv2.INTER_LINEAR
)

cv2.imwrite("test_resized.png", resized_back)
test_robustness("Resize and restore", resized_back)

# Test C: Gaussian blur
blurred = cv2.GaussianBlur(
    stego,
    (3, 3),
    0
)

cv2.imwrite("test_blurred.png", blurred)
test_robustness("Gaussian blur", blurred)

# Test D: Add mild pixel noise
rng = np.random.default_rng(42)

noise = rng.normal(
    0,
    2,
    stego.shape
)

noisy = np.clip(
    stego.astype(np.float32) + noise,
    0,
    255
).astype(np.uint8)

cv2.imwrite("test_noisy.png", noisy)
test_robustness("Mild Gaussian noise", noisy)

# -----------------------------
# 8. SAVE SUMMARY
# -----------------------------
with open("metrics_summary.txt", "w", encoding="utf-8") as file:
    file.write("DEEPDETECT IMAGE METRICS\n")
    file.write("========================\n")
    file.write(f"MSE: {mse:.6f}\n")
    file.write(f"PSNR: {psnr:.4f} dB\n")
    file.write(f"Embedded bits: {embedded_bits}\n")
    file.write(f"Payload rate: {capacity_bpp:.6f} bits/pixel\n")
    file.write(f"Capacity used: {capacity_percentage:.4f}%\n")
    file.write(f"Original BER: {ber:.6f}\n")

print("\nMetrics saved to metrics_summary.txt")
print("All tests completed.")
