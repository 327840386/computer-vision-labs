# texture_classification.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import gradio as gr


# Step 1: Dataset Preparation
def load_images(image_folder):
    images = []
    labels = []

    for label in ['grass', 'wood']:
        folder_path = os.path.join(image_folder, label)
        for image_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_file)
            print(f"Loading image: {img_path}")  # Print the image path to debug
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale

            # Check if the image was loaded correctly
            if img is None:
                print(f"Warning: Failed to load image: {img_path}")
                continue  # Skip this image

            # Resize to a fixed size (e.g., 128x128)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(0 if label == 'grass' else 1)  # Assign 0 to grass, 1 to wood

    return np.array(images), np.array(labels)


# Load dataset
image_folder = "/Users/luyuhao/Desktop/computer_vision/textures_dataset"  # Change this to your dataset path
images, labels = load_images(image_folder)

# Step 2: Stratified Data Splitting to Ensure Both Classes Are Represented

# Stratified split to ensure both classes are represented in the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42, stratify=labels)

# Check the label distribution in the training and testing sets
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)
print(f"Training labels distribution: {dict(zip(unique_train, counts_train))}")
print(f"Testing labels distribution: {dict(zip(unique_test, counts_test))}")

# Step 3: Feature Extraction


# GLCM Feature Extraction
def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, correlation, energy, homogeneity]


# LBP Feature Extraction
def extract_lbp_features(image, radius=1, n_points=8):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist


# Extract features for training and testing sets
glcm_features_train = np.array([extract_glcm_features(img) for img in X_train])
glcm_features_test = np.array([extract_glcm_features(img) for img in X_test])

lbp_features_train = np.array([extract_lbp_features(img) for img in X_train])
lbp_features_test = np.array([extract_lbp_features(img) for img in X_test])


# Step 4: Classification

# Train SVM on GLCM Features
svm_glcm = SVC(kernel='linear')
svm_glcm.fit(glcm_features_train, y_train)

# Train SVM on LBP Features
svm_lbp = SVC(kernel='linear')
svm_lbp.fit(lbp_features_train, y_train)

# Evaluate GLCM Model
y_pred_glcm = svm_glcm.predict(glcm_features_test)
accuracy_glcm = accuracy_score(y_test, y_pred_glcm)
conf_matrix_glcm = confusion_matrix(y_test, y_pred_glcm)

print(f"GLCM SVM Accuracy: {accuracy_glcm}")
print(f"GLCM Confusion Matrix:\n{conf_matrix_glcm}")

# Evaluate LBP Model
y_pred_lbp = svm_lbp.predict(lbp_features_test)
accuracy_lbp = accuracy_score(y_test, y_pred_lbp)
conf_matrix_lbp = confusion_matrix(y_test, y_pred_lbp)

print(f"LBP SVM Accuracy: {accuracy_lbp}")
print(f"LBP Confusion Matrix:\n{conf_matrix_lbp}")


# Step 5: Gradio Interface

def classify_image(image, method='GLCM'):
    print(f"Image type: {type(image)}, Image shape: {getattr(image, 'shape', 'None')}")  # Debugging output

    if image is None or image.size == 0:
        return "Invalid image. Please upload a valid image."

    # Preprocess the image (convert to grayscale)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (128, 128))

    if method == 'GLCM':
        features = np.array(extract_glcm_features(img)).reshape(1, -1)
        prediction = svm_glcm.predict(features)
    else:
        features = np.array(extract_lbp_features(img)).reshape(1, -1)
        prediction = svm_lbp.predict(features)

    return "Grass" if prediction[0] == 0 else "Wood"


# Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=[
        gr.Image(type="numpy", label="Upload Image"),
        gr.Radio(["GLCM", "LBP"], label="Method")
    ],
    outputs=gr.Textbox(label="Classification Result"),
    live=True
)

# Launch Gradio app
interface.launch(share=True)
