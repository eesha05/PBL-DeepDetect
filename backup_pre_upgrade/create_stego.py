import cv2

def hide_message(image_path, secret_message):
    image = cv2.imread(image_path)
    binary_message = ''.join(format(ord(i), '08b') for i in secret_message)
    binary_message += '1111111111111110'  # End delimiter
    
    data_index = 0
    rows, cols, _ = image.shape

    for row in range(rows):
        for col in range(cols):
            for channel in range(3):  # RGB channels
                if data_index < len(binary_message):
                    pixel = format(image[row][col][channel], '08b')
                    image[row][col][channel] = int(pixel[:-1] + binary_message[data_index], 2)
                    data_index += 1

    cv2.imwrite("stego.jpg", image)
    print("Stego image created as stego.jpg")

hide_message("clean.jpg", "Hidden secret message")
