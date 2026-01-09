import sys
import os

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import streamlit as st
from PIL import Image
import torch

# Import từ src
from src.dataset import get_transforms
from src.predict import predict_one_image
from src.model import SimpleCNNDog
from src.utils import (
    load_custom_css,
    render_header,
    render_sidebar,
    render_upload_section,
    render_result_section
)

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Are You A Dog? 🐕",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# MODEL FUNCTIONS
# ============================================

@st.cache_resource
def load_model_for_inference(model_path, device):
    """Load trained model (cached)"""
    try:
        model = SimpleCNNDog().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image):
    """Preprocess image for model"""
    transforms = get_transforms()
    image = transforms(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# ============================================
# MAIN APP
# ============================================

def main():
    # Load custom CSS
    load_custom_css()
    
    # Render header
    render_header()
    
    # Render sidebar and get settings
    render_sidebar()
    
    # Upload section
    uploaded_file = render_upload_section()
    
    if uploaded_file is not None:
        # Display uploaded image
        st.markdown("### Your Image")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.markdown("---")
        
        # Predict button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Analyze Image", use_container_width=True):
                with st.spinner("AI is thinking..."):
                    try:
                        # Load model
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = load_model_for_inference("models/checkpoints/model_weights.pth", device)
                        
                        if model is not None:
                            # Make prediction
                            pred, conf = predict_one_image(
                                model,
                                uploaded_file,
                                transform=get_transforms(),
                                device=device
                            )
                            
                            # Show results
                            st.markdown("---")
                            st.markdown("### Results")
                            render_result_section(pred, conf)
                            
                            # Success message
                            st.success("✅ Analysis complete!")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    else:
        pass


if __name__ == "__main__":
    main()
