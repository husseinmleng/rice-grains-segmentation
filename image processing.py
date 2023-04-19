import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import segmentation
import argparse

# Load the input image from the specified file path
def load_image(image_path):
    return cv2.imread(image_path)

# Display the provided image using Matplotlib
def show(image, x=30, y=7):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(x, y))
    plt.imshow(img)

# Preprocess the image to improve the quality of the detected contours
def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh_img = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3), np.uint8)
    clear_image = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=8)
    return clear_image

# Count the number of grains in the preprocessed image using flood fill
def count_grains(image):
    label_image = image.copy()
    label_count = 0
    rows, cols = label_image.shape
    for j in range(rows):
        for i in range(cols):
            pixel = label_image[j, i]
            if 255 == pixel:
                label_count += 1
                cv2.floodFill(label_image, None, (i, j), label_count)
    return label_image, label_count

# Detect the contours in the preprocessed image using OpenCV
def detect_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output_contour = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_contour, contours, -1, (0, 0, 255), 2)
    return output_contour, len(contours)

# Save the output image with the detected contours to the specified file path
def save_image(image, count, output_path):
    cv2.imwrite(output_path, image)
    print(f'Saved image with {count} detected contours at {output_path}')

# Parse command line arguments and call the appropriate functions
def main():
    parser = argparse.ArgumentParser(description='Detect contours in an image')
    parser.add_argument('--input_image', type=str, help='path to the input image file')
    parser.add_argument('--output_image', type=str, help='path to save the output image file')
    args = parser.parse_args()

    # Load the input image
    image = load_image(args.input_image)

    # Preprocess the image
    clear_image = preprocess_image(image)

    # Count the number of grains in the preprocessed image
    label_image, label_count = count_grains(clear_image)

    # Detect the contours in the preprocessed image
    output_image, contour_count = detect_contours(clear_image)

    # Save the output image with the detected contours
    save_image(output_image, contour_count, args.output_image)

# Only call the main function if the script is run directly
if __name__ == '__main__':
    main()