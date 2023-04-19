import cv2
import numpy as np
from skimage import feature, segmentation
import argparse

# Load the input image from the specified file path
def load_image(image_path):
    return cv2.imread(image_path)

# Display the provided image using OpenCV
def show(image, window_name='image'):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Preprocess the image to improve the quality of the detected contours
def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    ret, thresh_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresh_img

# Segment the binary image using watershed algorithm
def segment_image(image):
    distance = cv2.distanceTransform(image, cv2.DIST_L2, 5)
    local_max = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
    markers = cv2.connectedComponents(local_max.astype(np.uint8))[1]
    markers = markers+1
    markers[image==0] = 0
    segmented = segmentation.watershed(-distance, markers, mask=image)
    return segmented

# Count the number of grains in the segmented image
def count_grains(image):
    label_count = len(np.unique(image))-1
    return image, label_count

# Save the output image with the detected contours to the specified file path
def save_image(image, output_path):
    cv2.imwrite(output_path, image)
    print(f'Saved image with detected contours at {output_path}')

# Parse command line arguments and call the appropriate functions
def main():
    parser = argparse.ArgumentParser(description='Detect contours in an image')
    parser.add_argument('--input_image', type=str, help='path to the input image file')
    parser.add_argument('--output_image', type=str, help='path to save the output image file')
    args = parser.parse_args()

    # Load the input image
    image = load_image(args.input_image)

    # Preprocess the image
    binary_image = preprocess_image(image)

    # Segment the binary image using watershed algorithm
    labeled_image = segment_image(binary_image)

    # Count the number of grains in the segmented image
    labeled_image, grain_count = count_grains(labeled_image)

    # Output the three images
    cv2.imwrite('original_img.jpg', image)
    cv2.imwrite(args.input_image+'binary_img.jpg', binary_image)
    cv2.imwrite(args.output_image, labeled_image)

    # Print the grain count
    print(f'Number of grains detected: {grain_count}')

# Only call the main function if the script is run directly
if __name__ == '__main__':
    main()
