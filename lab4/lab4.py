import torch
import torchvision.transforms as T
from PIL import Image
import cv2
from torchvision import models
import numpy as np
import gradio as gr


def segment_person(image_path):
    try:
        # Load pre-trained model (e.g., DeepLabV3)
        model = models.segmentation.deeplabv3_resnet101(pretrained=True).eval()

        # Preprocess the image
        input_image = Image.open(image_path)
        preprocess = T.Compose([
            T.Resize((520, 520)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0)

        # Forward pass to obtain segmentation mask
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
            output_predictions = output.argmax(0)

        # Create a mask and resize to match original image dimensions
        mask = output_predictions.byte().cpu().numpy()
        original_image = cv2.imread(image_path)
        original_height, original_width = original_image.shape[:2]

        # Resize mask to match original image size
        person_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        # Create RGBA image with transparency
        image_np = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGBA)
        image_np[:, :, 3] = person_mask.astype(np.uint8) * 255

        # Save result with transparency
        segmented_path = "/Users/luyuhao/Desktop/computer_vision/lab4/segmented_person.png"
        Image.fromarray(image_np).save(segmented_path)
        return segmented_path

    except Exception as e:
        print(f"Error during segmentation: {e}")
        return None


def overlay_image(background, overlay, position):
    """
    Overlay an image onto a background image.
    :param background: Background image, must be an OpenCV image (BGR format).
    :param overlay: The image to overlay, must have an alpha channel (BGRA format).
    :param position: (x, y) The position where the overlay image is placed on the background.
    """
    x, y = position
    h, w = overlay.shape[:2]

    # Ensure the position is valid and does not exceed the background image boundaries
    if y + h > background.shape[0] or x + w > background.shape[1]:
        raise ValueError("The overlay image exceeds the background boundaries.")

    # Extract the region of interest (ROI) on the background image
    roi = background[y:y+h, x:x+w]

    # Create a mask
    mask = overlay[:, :, 3] != 0  # Only overlay where the alpha channel is not 0

    # Use the mask to overlay the content onto the background image's corresponding region
    roi[mask] = overlay[:, :, :3][mask]

    # Write the updated part back to the original background
    background[y:y+h, x:x+w] = roi


def insert_into_stereo(left_image_path, right_image_path, person_image_path, depth_level="medium"):
    try:
        # Load the left and right images
        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)

        # Check if the images were successfully loaded
        if left_image is None:
            raise FileNotFoundError(f"Could not load left image from path: {left_image_path}")
        if right_image is None:
            raise FileNotFoundError(f"Could not load right image from path: {right_image_path}")

        # Load the segmented person image and convert it to OpenCV format (BGRA)
        person_image = Image.open(person_image_path)
        person_image_cv = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGBA2BGRA)

        # Define the disparity for different depth levels
        disparity = {"close": 30, "medium": 15, "far": 5}
        d = disparity[depth_level]

        # Define the insertion positions
        left_pos = (100, 100)  # Position in the left image
        right_pos = (100 + d, 100)  # Position in the right image

        # Overlay the person onto the left and right images
        overlay_image(left_image, person_image_cv, left_pos)
        overlay_image(right_image, person_image_cv, right_pos)

        # Save the modified stereo images
        left_output_path = '/Users/luyuhao/Desktop/computer_vision/lab4/stereo_left.png'
        right_output_path = '/Users/luyuhao/Desktop/computer_vision/lab4/stereo_right.png'
        cv2.imwrite(left_output_path, left_image)
        cv2.imwrite(right_output_path, right_image)

        return left_output_path, right_output_path
    except Exception as e:
        print(f"Error during stereo insertion: {e}")
        return None, None


def adjust_depth_insertion(left_image_path, right_image_path, person_image_path, depth_level="medium"):
    # Load the left and right images
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # Load the segmented person image and convert it to OpenCV format (BGRA)
    person_image = Image.open(person_image_path)
    person_image_cv = cv2.cvtColor(np.array(person_image), cv2.COLOR_RGBA2BGRA)

    # Define the disparity values for each depth level
    disparity_map = {
        "close": 50,
        "medium": 25,
        "far": 10
    }
    disparity = disparity_map[depth_level]

    # Define the insertion positions in the left and right images
    left_position = (100, 150)  # Position in the left image
    right_position = (left_position[0] + disparity, left_position[1])  # Position in the right image

    # Overlay the person onto the left and right images
    overlay_image(left_image, person_image_cv, left_position)
    overlay_image(right_image, person_image_cv, right_position)

    # Save the stereo images with adjusted depth
    left_output_path = '/Users/luyuhao/Desktop/computer_vision/lab4/stereo_left_adjusted.png'
    right_output_path = '/Users/luyuhao/Desktop/computer_vision/lab4/stereo_right_adjusted.png'
    cv2.imwrite(left_output_path, left_image)
    cv2.imwrite(right_output_path, right_image)

    return left_output_path, right_output_path


# Example usage
left_image_path = '/Users/luyuhao/Desktop/computer_vision/lab4/left_image.png'   # Path to left stereo image
right_image_path = '/Users/luyuhao/Desktop/computer_vision/lab4/right_image.png' # Path to right stereo image
person_image_path = '/Users/luyuhao/Desktop/computer_vision/lab4/person.png' # Path to segmented person image

# Adjust depth by choosing 'close', 'medium', or 'far'
adjusted_left, adjusted_right = adjust_depth_insertion(left_image_path, right_image_path, person_image_path, depth_level="close")


def create_anaglyph(left_image_path, right_image_path):
    try:
        # Read the left and right images
        left_image = cv2.imread(left_image_path)
        right_image = cv2.imread(right_image_path)

        # Check if the images were successfully loaded
        if left_image is None:
            raise FileNotFoundError(f"Could not load left image from path: {left_image_path}")
        if right_image is None:
            raise FileNotFoundError(f"Could not load right image from path: {right_image_path}")

        # Resize the right image to match the left image
        if left_image.shape != right_image.shape:
            right_image = cv2.resize(right_image, (left_image.shape[1], left_image.shape[0]))

        # Ensure both images have the same depth (convert to uint8)
        left_image = cv2.convertScaleAbs(left_image)
        right_image = cv2.convertScaleAbs(right_image)

        # Extract color channels to create the anaglyph image
        r = left_image[:, :, 2]  # Extract the red channel from the left image
        g = right_image[:, :, 1]  # Extract the green channel from the right image
        b = right_image[:, :, 0]  # Extract the blue channel from the right image

        # Create the anaglyph image
        anaglyph = cv2.merge((b, g, r))
        anaglyph_path = '/Users/luyuhao/Desktop/computer_vision/anaglyph.png'
        cv2.imwrite(anaglyph_path, anaglyph)

        return anaglyph_path
    except Exception as e:
        print(f"Error during anaglyph creation: {e}")
        return None


def gradio_interface(person_image, depth):
    segmented_person = segment_person(person_image)
    stereo_left, stereo_right = insert_into_stereo('/Users/luyuhao/Desktop/computer_vision/lab4/left_image.png',
                                                   '/Users/luyuhao/Desktop/computer_vision/lab4/right_image.png', segmented_person, depth)
    anaglyph_image = create_anaglyph(stereo_left, stereo_right)
    return anaglyph_image


iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Image(type="filepath", label="Upload an image of a person"),
        gr.Radio(choices=["close", "medium", "far"], label="Depth Level")
    ],
    outputs=gr.Image(type="filepath"),
    title="3D Image Composer"
)

iface.launch(share=True)
