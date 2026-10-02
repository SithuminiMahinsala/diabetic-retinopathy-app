import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import requests

st.set_page_config(page_title="DR Stage Detector", layout="centered")

st.title("Diabetic Retinopathy Clinical Triage")
st.write("Automated screening system powered by ResNet-50 and CLAHE contrast enhancement.")

device = torch.device("cpu")
MODEL_PATH = "best_retina_model.pth"


MODEL_URL = "https://github.com/SithuminiMahinsala/diabetic-retinopathy-app/releases/download/v1.0/best_retina_model.pth"

#  Automatic weight downloader
def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model weights (~95MB), please wait..."):
            response = requests.get(MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

# Load Model Architecture & Weights
@st.cache_resource
def load_trained_model():
    download_model_if_missing()
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 5)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_trained_model()

class_names = {0: 'No DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative DR'}

val_transforms = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def crop_retina_circle(image, tol=7):
    if image.ndim == 2:
        mask = image > tol
        return image[np.ix_(mask.any(1), mask.any(0))]
    elif image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = gray > tol
        check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return image
        else:
            img1 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            return np.stack([img1, img2, img3], axis=-1)

def preprocess_retina(img, img_size=224):
    cropped = crop_retina_circle(img)
    resized = cv2.resize(cropped, (img_size, img_size))
    lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)

uploaded_file = st.file_uploader("Upload a Retinal Fundus Scan (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    raw_pil = Image.open(uploaded_file).convert("RGB")
    raw_np = np.array(raw_pil)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_np, caption="Original Upload", use_container_width=True)
        
    processed = preprocess_retina(raw_np)
    with col2:
        st.image(processed, caption="Preprocessed (Crop + CLAHE)", use_container_width=True)
        
    tensor_input = val_transforms(image=processed)['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor_input)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))

    st.subheader(f"Diagnosis: **{class_names[pred_idx]}** ({probs[pred_idx]*100:.1f}% confidence)")
    
    for i in range(5):
        st.write(f"**{class_names[i]}**")
        st.progress(float(probs[i]))
