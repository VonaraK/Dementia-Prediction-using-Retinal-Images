import streamlit as st
import cv2
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.spatial.distance import euclidean
from scipy.stats import linregress
import joblib  # For loading models
from tensorflow.keras.models import load_model  # For segmentation model
from albumentations import Compose, HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast, ElasticTransform

# Import the Unet Attention model from your custom module
import sys
sys.path.append('unet-model')
from model import attentionunet

# Load the pre-trained segmentation model
input_shape = (256, 256, 1)

def load_segmentation_model(model_path):
    model = attentionunet(input_shape=input_shape)
    model.load_weights(model_path)
    return model

# Load the XGBoost model
def load_xgb_model(model_path='xgb_model.pkl'):
    return joblib.load(model_path)

# Load the scaler
def load_scaler(scaler_path='scaler.pkl'):
    return joblib.load(scaler_path)

# Define augmentation pipeline
def augment_image_realistic(image):
    augmentation_pipeline = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ElasticTransform(alpha=1, sigma=50, p=0.3)
    ])
    augmented = augmentation_pipeline(image=image)
    return augmented['image']

# Preprocessing function
def preprocess_image(image, target_size=256):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(image)
    img_blur = cv2.GaussianBlur(img_clahe, (5, 5), 0)
    img_padded = pad_to_square(img_blur, target_size)
    img_normalized = img_padded / 255.0
    img_normalized = np.expand_dims(img_normalized, axis=(0, -1))
    return img_normalized

def pad_to_square(img, target_size):
    height, width = img.shape[:2]
    max_dim = max(height, width)
    square_img = np.zeros((max_dim, max_dim), dtype=img.dtype)
    y_offset = (max_dim - height) // 2
    x_offset = (max_dim - width) // 2
    square_img[y_offset:y_offset + height, x_offset:x_offset + width] = img
    return cv2.resize(square_img, (target_size, target_size))

# Vascular Metrics Calculations
def arc_length_chord_length_ratio(vessel_segment):
    coords = np.column_stack(np.where(vessel_segment))
    arc_length = np.sum([euclidean(coords[i], coords[i+1]) for i in range(len(coords) - 1)])
    chord_length = euclidean(coords[0], coords[-1])
    return arc_length / chord_length if chord_length > 0 else 0

# Fractal Dimension Calculation
def calculate_fractal_dimension(skeleton, box_sizes=[2, 4, 8, 16, 32, 64]):
    try:
        if np.sum(skeleton) < 10:  # Ensure the skeleton is not too sparse
            return None
        box_counts = []
        for box_size in box_sizes:
            resized_img = cv2.resize(
                skeleton.astype(np.uint8),
                (skeleton.shape[1] // box_size, skeleton.shape[0] // box_size),
                interpolation=cv2.INTER_NEAREST
            )
            box_counts.append(np.sum(resized_img > 0))
        log_box_sizes = np.log(np.array(box_sizes))
        log_box_counts = np.log(np.array(box_counts))
        slope, _, _, _, _ = linregress(log_box_sizes, log_box_counts)
        return -slope
    except:
        return None

def calculate_cra_or_crv(diameters, constant):
    diameters = sorted(diameters, reverse=True)[:6]
    while len(diameters) > 1:
        combined = constant * np.sqrt(diameters[0]**2 + diameters[-1]**2)
        diameters = diameters[1:-1]
        diameters.append(combined)
    return diameters[0] if diameters else None

# Image Processing
def process_image(image, model):
    preprocessed_img = preprocess_image(image)
    segmented_output = model.predict(preprocessed_img)[0, :, :, 0]
    skeleton = skeletonize(segmented_output > 0.5)
    labeled_skeleton = label(skeleton, connectivity=2)

    tortuosity_list = []
    diameters = []
    for i in range(1, labeled_skeleton.max() + 1):
        vessel_segment = (labeled_skeleton == i)
        tortuosity = arc_length_chord_length_ratio(vessel_segment)
        tortuosity_list.append(tortuosity)
        coords = np.column_stack(np.where(vessel_segment))
        if len(coords) > 1:
            max_diameter = np.linalg.norm(coords[-1] - coords[0])
            diameters.append(max_diameter)

    pixel_size = 4
    diameters_in_micrometers = [d * pixel_size for d in diameters]
    CRAE = calculate_cra_or_crv(diameters_in_micrometers, constant=0.88)
    CRVE = calculate_cra_or_crv(diameters_in_micrometers, constant=0.95)
    AVR = CRAE / CRVE if CRAE and CRVE else None
    fractal_dimension = calculate_fractal_dimension(skeleton)

    return {
        'Mean Tortuosity': np.mean(tortuosity_list) if tortuosity_list else None,
        'CRAE': CRAE,
        'CRVE': CRVE,
        'AVR': AVR,
        'Fractal Dimension': fractal_dimension,
    }

# Predict Condition using XGBoost model
def predict_condition(extracted_params, xgb_model, scaler):
    input_features = np.array([
        extracted_params['Mean Tortuosity'],
        extracted_params['CRAE'],
        extracted_params['CRVE'],
        extracted_params['AVR'],
        extracted_params['Fractal Dimension']
    ]).reshape(1, -1)

    if None in input_features[0] or np.isnan(input_features).any():
        return "Insufficient Data for Prediction"

    scaled_features = scaler.transform(input_features)
    prediction = xgb_model.predict(scaled_features)
    if prediction == 1:
        return "Alzheimer's Disease (AD)"
    elif prediction == 2:
        return "Vascular Dementia (VaD)"
    else:
        return "Healthy"

# Streamlit UI
def main():
    st.title("Retinal Image Analysis for Dementia Prediction")

    uploaded_right_eye = st.file_uploader("Upload Right Eye Image", type=["jpg", "png"])
    uploaded_left_eye = st.file_uploader("Upload Left Eye Image", type=["jpg", "png"])

    if uploaded_right_eye and uploaded_left_eye:
        right_eye_image = cv2.imdecode(np.frombuffer(uploaded_right_eye.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        left_eye_image = cv2.imdecode(np.frombuffer(uploaded_left_eye.read(), np.uint8), cv2.IMREAD_GRAYSCALE)

        model = load_segmentation_model('unet-model/Trained models/retina_attentionUnet_150epochs.hdf5')
        xgb_model = load_xgb_model('xgb_model.pkl')
        scaler = load_scaler('scaler.pkl')

        right_eye_params = process_image(right_eye_image, model)
        left_eye_params = process_image(left_eye_image, model)

        right_eye_condition = predict_condition(right_eye_params, xgb_model, scaler)
        left_eye_condition = predict_condition(left_eye_params, xgb_model, scaler)

        st.subheader("Right Eye Analysis")
        st.table(pd.DataFrame([right_eye_params]))
        st.subheader(f"Prediction: {right_eye_condition}")

        st.subheader("Left Eye Analysis")
        st.table(pd.DataFrame([left_eye_params]))
        st.subheader(f"Prediction: {left_eye_condition}")
    else:
        st.warning("Please upload both right and left eye images")

if __name__ == "__main__":
    main()
