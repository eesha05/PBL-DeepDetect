import cv2
import matplotlib.pyplot as plt

img = cv2.imread('stego.jpg', 0)

plt.figure(figsize=(10,8))

for i in range(8):
    bit_plane = (img >> i) & 1
    plt.subplot(2,4,i+1)
    plt.imshow(bit_plane, cmap='gray')
    plt.title(f'Bit Plane {i}')
    plt.axis('off')

plt.tight_layout()
plt.show()