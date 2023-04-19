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
    return segmented.astype(np.uint8)

# Count the number of grains in the segmented image
def count_grains(image):
    label_count = len(np.unique(image))-1
    return image, label_count

def color_instances(image):
    # Generate a random color map with a unique color for each instance
    rng = np.random.RandomState(0)
    colors = rng.randint(0, 256, size=(np.max(image)+1, 3), dtype=np.uint8)
    colors[0, :] = 0 # Set the background color to black
    color_map = colors[image]

    # Create a mask for each instance in the labeled image
    masks = [(image == i).astype(np.uint8) for i in range(1, np.max(image)+1)]

    # Color each instance with a unique color
    colored_image = np.zeros_like(color_map)
    for i, mask in enumerate(masks):
        colored_mask = cv2.bitwise_and(color_map, color_map, mask=mask)
        colored_mask = colored_mask.astype(colored_image.dtype) # Cast colored_mask to the same data type as colored_image
        colored_image = (colored_image + colored_mask).astype(np.uint8)

    return colored_image

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

    # Color the instances of the labeled image
    colored_image = color_instances(labeled_image)

    # Concatenate the three images horizontally
    output_image = np.concatenate((image, cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), colored_image), axis=1)

    # Save the output image
    cv2.imwrite(args.output_image, cv2.convertScaleAbs(output_image))

    # Print the grain count
    print(f'Number of grains detected: {grain_count}')

# Only call the main function if the script is run directly
if __name__ == '__main__':
    main()