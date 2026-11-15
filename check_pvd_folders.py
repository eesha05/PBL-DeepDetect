import os

BASE_PATH = r"C:\Users\eesha\Downloads\archive (2)\Stego-pvd-dataset\test"

print("Folders inside the test directory:\n")

if not os.path.exists(BASE_PATH):
    print("The path does not exist.")
else:
    for item in os.listdir(BASE_PATH):
        full_path = os.path.join(BASE_PATH, item)

        if os.path.isdir(full_path):
            print("Folder:", item)
        else:
            print("File:", item)