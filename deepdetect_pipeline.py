import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
CLEAN_DIR = "data/clean"
STEGO_DIR = "data/stego"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
def load_and_preprocess(image_path, image_size=(224, 224)):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    resized_image = cv2.resize(image_rgb, image_size)

    normalized_image = resized_image.astype(np.float32) / 255.0

    grayscale_image = cv2.cvtColor(
        resized_image,
        cv2.COLOR_RGB2GRAY
    )

    return resized_image, normalized_image, grayscale_image


def sobel_edges(grayscale_image):
    sobel_x = cv2.Sobel(
        grayscale_image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        grayscale_image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    gradient_magnitude = np.sqrt(
        sobel_x ** 2 + sobel_y ** 2
    )

    gradient_magnitude = cv2.normalize(
        gradient_magnitude,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return gradient_magnitude.astype(np.uint8)

def laplacian_edges(grayscale_image):
    laplacian = cv2.Laplacian(
        grayscale_image,
        cv2.CV_64F
    )

    laplacian = np.absolute(laplacian)

    laplacian = cv2.normalize(
        laplacian,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return laplacian.astype(np.uint8)

def canny_edges(grayscale_image):
    return cv2.Canny(
        grayscale_image,
        threshold1=50,
        threshold2=150
    )

def calculate_noise_residual(grayscale_image):
    blurred_image = cv2.GaussianBlur(
        grayscale_image,
        (5, 5),
        0
    )

    noise_residual = cv2.absdiff(
        grayscale_image,
        blurred_image
    )

    return noise_residual

def calculate_edge_density(edge_image):
    edge_pixels = np.count_nonzero(edge_image)

    total_pixels = edge_image.shape[0] * edge_image.shape[1]

    return edge_pixels / total_pixels

def extract_features(grayscale_image):
    sobel_image = sobel_edges(grayscale_image)
    laplacian_image = laplacian_edges(grayscale_image)
    canny_image = canny_edges(grayscale_image)
    noise_image = calculate_noise_residual(grayscale_image)

    features = {
        "sobel_mean": float(np.mean(sobel_image)),
        "sobel_std": float(np.std(sobel_image)),

        "laplacian_mean": float(np.mean(laplacian_image)),
        "laplacian_variance": float(np.var(laplacian_image)),

        "canny_edge_density": float(
            calculate_edge_density(canny_image)
        ),

        "noise_mean": float(np.mean(noise_image)),
        "noise_std": float(np.std(noise_image))
    }

    return features

def generate_feature_csv():

    all_features = []

    image_groups = [
        (CLEAN_DIR, 0, "clean"),
        (STEGO_DIR, 1, "stego")
    ]

    for folder, label, image_type in image_groups:

        if not os.path.exists(folder):
            print(f"Folder not found: {folder}")
            continue

        for filename in sorted(os.listdir(folder)):

            if not filename.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            ):
                continue

            image_path = os.path.join(folder, filename)

            try:
                _, _, grayscale_image = load_and_preprocess(
                    image_path
                )

                features = extract_features(grayscale_image)

                features["filename"] = filename
                features["image_type"] = image_type
                features["label"] = label

                all_features.append(features)

                print(f"Processed: {image_path}")

            except Exception as error:
                print(
                    f"Error processing {image_path}: {error}"
                )

    if len(all_features) == 0:
        print("No images were found.")
        return

    dataframe = pd.DataFrame(all_features)

    output_path = os.path.join(
        OUTPUT_DIR,
        "deepdetect_edge_features.csv"
    )

    dataframe.to_csv(output_path, index=False)

    print("\nFeature CSV created successfully.")
    print(f"Saved at: {output_path}")
    print("\nClass distribution:")
    print(dataframe["image_type"].value_counts())

def display_analysis(image_path):

    resized_image, _, grayscale_image = load_and_preprocess(
        image_path
    )

    sobel_image = sobel_edges(grayscale_image)
    laplacian_image = laplacian_edges(grayscale_image)
    canny_image = canny_edges(grayscale_image)
    noise_image = calculate_noise_residual(grayscale_image)

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(resized_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(grayscale_image, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(sobel_image, cmap="gray")
    plt.title("Sobel Edges")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(laplacian_image, cmap="gray")
    plt.title("Laplacian Edges")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(canny_image, cmap="gray")
    plt.title("Canny Edges")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(noise_image, cmap="gray")
    plt.title("Noise Residual")
    plt.axis("off")

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "edge_analysis_result.png"
    )

    plt.savefig(output_path, dpi=300)

    plt.show()

    print(f"Saved visualization at: {output_path}")

if __name__ == "__main__":

    print("Starting DeepDetect pipeline...\n")

    generate_feature_csv()

    sample_image = None

    for filename in sorted(os.listdir(CLEAN_DIR)):

        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        ):
            sample_image = os.path.join(
                CLEAN_DIR,
                filename
            )
            break

    if sample_image is not None:
        display_analysis(sample_image)
    else:
        print("No clean sample image found.")

    print("\nDeepDetect pipeline completed.")