import cv2
import numpy as np
import os
def segment_rice_grains(image_path,output_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Perform morphological operations (dilation followed by erosion) to close gaps and remove small artifacts
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size and aspect ratio to remove false positives
    filtered_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Set appropriate thresholds for size and aspect ratio based on your requirements
        if 100 < area < 5000 and 0.5 < aspect_ratio < 2:
            filtered_contours.append(cnt)

    # Draw contours on the original image
    result = cv2.drawContours(image, filtered_contours, -1, (0, 255, 0), 2)

    # Display the results
    # cv2.imshow('Original Image', image)
    cv2.imwrite(output_path+'out.jpg', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Replace 'rice_grains.jpg' with your image file
path = "/media/huusein/My Learning/Jupyter NoteBooks/Freelancing/Rice grains segmentation/Version 2/"
segment_rice_grains(path+'img.jpg',path)