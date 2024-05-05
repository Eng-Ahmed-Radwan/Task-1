import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def display_image_and_histograms(image, title):
    """Displays an image with its color channel histograms."""
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])

    plt.figure(figsize=(15, 6))

    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.plot(hist_b, color='blue')
    plt.title('Blue Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 4, 3)
    plt.plot(hist_g, color='green')
    plt.title('Green Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.subplot(2, 4, 4)
    plt.plot(hist_r, color='red')
    plt.title('Red Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')

    plt.tight_layout()


def adjust_contrast(image, alpha):
    """Adjusts the contrast of an image.

    Args:
        image: The input image as a NumPy array.
        alpha: A value to multiply the image pixels. Values greater than 1 increase contrast,
            values between 0 and 1 decrease contrast.

    Returns:
        A new image with adjusted contrast.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)


def adjust_brightness(image, beta):
    """Adjusts the brightness of an image.

    Args:
        image: The input image as a NumPy array.
        beta: A value to add or subtract to the image pixels. Positive values increase brightness,
            negative values decrease brightness.

    Returns:
        A new image with adjusted brightness.
    """
    return cv2.convertScaleAbs(image, alpha=1.0, beta=beta)


def main():
    """Prompts the user for an image path, loads the image, performs operations, and displays results."""

    # Step 1: Ask user to input image path
    while True:
        image_path = input("Enter the path to the image: ")
        if not image_path:
            print("Please enter a valid image path.")
            continue

        # Check if the file exists
        if not os.path.isfile(image_path):
            print(f"Error: File '{image_path}' does not exist. Please try again.")
            continue

        break  # Exit the loop if a valid path is provided

    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if image is None:
        print(f"Error loading image from {image_path}. Check the file path and permissions.")
        return

    # Generate random values for contrast and brightness adjustment
    random_contrast = random.uniform(0.5, 2.0)  # Random value between 0.5 and 2.0
    random_brightness = random.randint(-100, 100)  # Random integer value between -100 and 100

    # Display the original image with histograms
    display_image_and_histograms(image.copy(), 'Original Image')

    # Example for contrast adjustment
    contrast_adjusted_image = adjust_contrast(image.copy(), random_contrast)
    display_image_and_histograms(contrast_adjusted_image, f'Image with Adjusted Contrast (Factor: {random_contrast:.2f})')

    # Example for brightness adjustment
    brightness_adjusted_image = adjust_brightness(image.copy(), random_brightness)
    display_image_and_histograms(brightness_adjusted_image, f'Image with Adjusted Brightness (Beta: {random_brightness})')

    plt.show()
    print("All operations completed.")


if __name__ == "__main__":
    main()
