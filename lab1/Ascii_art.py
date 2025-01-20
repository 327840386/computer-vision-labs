import cv2
import gradio as gr
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Define a list of ASCII characters to represent pixel intensities
ASCII_CHARS = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.']


def convert_to_grayscale(image):
    """Convert a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_edge_detection(grayscale_image):
    """Apply Gaussian Blur followed by Canny edge detection."""
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)  # Blur the image to reduce noise
    return cv2.Canny(blurred_image, threshold1=100, threshold2=200)  # Detect edges


def pixel_to_ascii(image, ascii_chars=ASCII_CHARS):
    """Map pixel values to ASCII characters."""
    ascii_image = []  # List to hold ASCII lines
    height, width = image.shape  # Get dimensions of the image

    for i in range(height):
        line = []  # Temporary list for each line of ASCII characters
        for j in range(width):
            pixel_value = image[i, j]  # Get pixel value
            char_index = min(pixel_value // (256 // len(ascii_chars)), len(ascii_chars) - 1)  # Map to ASCII index
            line.append(ascii_chars[char_index])  # Append corresponding ASCII character
        ascii_image.append(''.join(line))  # Join characters to form a line

    return '\n'.join(ascii_image)  # Return the ASCII art as a string


def mean_squared_error(original_image, ascii_image):
    """Calculate Mean Squared Error between original image and ASCII image."""
    ascii_image = ascii_image.astype(np.uint8) * 255  # Convert ASCII image to 8-bit values
    mse = np.mean((original_image - ascii_image) ** 2)  # Calculate MSE
    return mse


def psnr(original_image, ascii_image):
    """Calculate Peak Signal-to-Noise Ratio between original image and ASCII image."""
    mse = mean_squared_error(original_image, ascii_image)  # Get MSE
    if mse == 0:
        return float('inf')  # Avoid division by zero
    return 20 * np.log10(255.0 / np.sqrt(mse))  # Calculate PSNR


def performance_metric(original_image, ascii_image):
    """Calculate performance metrics: SSIM, MSE, and PSNR."""
    ssim_value = ssim(original_image, ascii_image)  # Calculate Structural Similarity Index
    mse_value = mean_squared_error(original_image, ascii_image)  # Calculate MSE
    psnr_value = psnr(original_image, ascii_image)  # Calculate PSNR

    return ssim_value, mse_value, psnr_value  # Return the calculated metrics


def image_to_ascii(image):
    """Process the image and convert it to ASCII art."""
    new_width = 300  # Set the desired width for resizing
    aspect_ratio = image.shape[0] / image.shape[1]  # Maintain aspect ratio
    new_height = int(aspect_ratio * new_width)  # Calculate new height
    image = cv2.resize(image, (new_width, new_height))  # Resize image

    grayscale_image = convert_to_grayscale(image)  # Convert to grayscale
    edges = apply_edge_detection(grayscale_image)  # Apply edge detection
    ascii_art = pixel_to_ascii(edges)  # Convert edges to ASCII art

    # Write ASCII art to a text file
    with open('ascii_art.txt', 'w') as file:
        file.write(ascii_art)

    # Calculate performance metrics
    ssim_value, mse_value, psnr_value = performance_metric(grayscale_image, edges)

    return ascii_art, 'ascii_art.txt', {'SSIM': ssim_value, 'MSE': mse_value, 'PSNR': psnr_value}  # Return ASCII art and metrics


# CSS styles for Gradio interface
css = """
textarea {
    font-family: 'Courier New', monospace;
    font-size: 1.2px;
    letter-spacing: 0.6px;
    line-height: 1.1;
"""

# Define Gradio interface for user interaction
interface = gr.Interface(
    fn=image_to_ascii,  # Function to be called
    inputs="image",  # Input type
    outputs=[gr.Textbox(type="text", label="ASCII"), 
             gr.File(label="Download ASCII Art Text"), 
             gr.JSON(label="Performance Metrics")],  # Output types
    title="ASCII Art",  # Title of the interface
    description="Convert your images to ASCII art.",  # Description of the interface
    css=css  # Apply CSS styles
)

# Launch the Gradio interface and allow sharing
interface.launch(share=True)
