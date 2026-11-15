import os
import shutil

# Stego image folder
STEGO_SOURCE = r"C:\Users\eesha\Downloads\archive (2)\Stego-pvd-dataset\test\stegoTest"

# Change "cleanTest" after checking the actual clean-image folder name
CLEAN_SOURCE = r"C:\Users\eesha\Downloads\archive (2)\Stego-pvd-dataset\test\cleanTest"

# Destination folders in your DeepDetect project
CLEAN_DEST = "data/clean"
STEGO_DEST = "data/stego"

# Number of images to copy from each class
NUMBER_OF_IMAGES = 100

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff"
)


def get_image_files(folder_path):
    """
    Returns sorted image filenames from a folder.
    """

    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return []

    image_files = [
        filename
        for filename in os.listdir(folder_path)
        if filename.lower().endswith(VALID_EXTENSIONS)
    ]

    image_files.sort()

    return image_files


def copy_images(source_folder, destination_folder, prefix):
    """
    Copies the first 100 images from the source folder.
    """

    os.makedirs(destination_folder, exist_ok=True)

    image_files = get_image_files(source_folder)

    if len(image_files) == 0:
        print(f"No images found in: {source_folder}")
        return

    selected_images = image_files[:NUMBER_OF_IMAGES]

    for index, filename in enumerate(selected_images, start=1):

        source_path = os.path.join(source_folder, filename)

        extension = os.path.splitext(filename)[1]

        destination_filename = f"{prefix}_{index:03d}{extension}"

        destination_path = os.path.join(
            destination_folder,
            destination_filename
        )

        shutil.copy2(source_path, destination_path)

        print(f"Copied {index}/{len(selected_images)}: {filename}")

    print(
        f"\nCopied {len(selected_images)} images "
        f"to {destination_folder}"
    )


print("Copying clean images...")
copy_images(CLEAN_SOURCE, CLEAN_DEST, "clean")

print("\nCopying stego images...")
copy_images(STEGO_SOURCE, STEGO_DEST, "stego")

print("\nDataset preparation completed.")