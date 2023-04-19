## Grain Contour Detection Script

This script is used to automatically detect the contours of grains in an image using computer vision techniques. The script takes an input image file and outputs another image file with detected contours over it.

### Requirements

- Python 3.6 or later
- OpenCV Python package
- NumPy Python package
- Matplotlib Python package
- scikit-image Python package

You can install the required packages using pip:

```
pip install opencv-python numpy matplotlib scikit-image
```

### Usage

To use the script, run it from the command line with the following arguments:

```
python grain_contour_detection.py input_image output_image
```

- `input_image`: Path to the input image file.
- `output_image`: Path to save the output image file with detected contours.

### How it works

The script works by first loading the input image using OpenCV. The image is then preprocessed to improve the quality of the detected contours. This involves converting the image to grayscale, applying a binary threshold, and using morphological operations to remove noise and fill gaps in the image.

Next, the number of grains in the preprocessed image is counted using flood fill. The flood fill algorithm is used to label each grain in the image with a unique ID.

Finally, the contours of the grains are detected using OpenCV. The detected contours are then drawn onto a copy of the preprocessed image in red and saved as the output image.

### Conclusion

The grain contour detection script provides a fast and automated way to detect the contours of grains in an image. This can be useful in various applications, such as quality control in the food industry or analysis of geological samples.