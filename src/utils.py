# ============================================
# STREAMLIT UI UTILITIES
# ============================================

import streamlit as st

def load_custom_css():
    """Load custom CSS for Streamlit app"""
    st.markdown("""
        <style>
        /* Main title styling */
        .main-title {
            text-align: center;
            font-size: 3.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle {
            text-align: center;
            font-size: 1.2rem;
            color: #666;
            margin-bottom: 2rem;
        }
        
        /* Result card styling */
        .result-card {
            padding: 2rem;
            border-radius: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .result-card-negative {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        
        .result-label {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
        }
        
        .result-confidence {
            font-size: 1.2rem;
            margin-top: 0.5rem;
            opacity: 0.9;
        }
        
        /* Info box */
        .info-box {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f0f2f6;
            border-left: 4px solid #667eea;
            margin: 1rem 0;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 3rem;
            font-weight: bold;
        }
        
        /* Upload section */
        .upload-section {
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }
        
        /* Stats card */
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render main header"""
    st.markdown('<h1 class="main-title">🐕 Are You A Dog?</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upload an image and let AI tell you if it\'s a dog or not!</p>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar():
    """Render sidebar with info and settings"""
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f436.png", width=80)
        st.title("About")
        st.info("""
        ### 🎯 What is this?
        This app uses deep learning to detect if an image contains a dog.
        
        ### 🚀 How to use:
        1. Upload an image (JPG, PNG)
        2. Wait for the magic ✨
        3. See the results!
        
        ### 🧠 Model Info:
        - Architecture: CNN
        - Trained on: Dog vs Cat dataset, Selfies dataset, Random images
        - Accuracy: ~85%
        """)
        
        st.markdown("---")
        st.caption("Made with ❤️ using Streamlit")


def render_upload_section():
    """Render upload section with nice styling"""
    st.markdown("### 📤 Upload Your Image")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of an animal"
        )
        
        if uploaded_file is None:
            st.markdown("""
                <div class="info-box">
                    <strong>💡 Tips for best results:</strong><br>
                    • Use clear, well-lit images<br>
                    • Make sure the animal is visible<br>
                    • JPG or PNG format works best
                </div>
            """, unsafe_allow_html=True)
    
    return uploaded_file


def render_result_section(pred, conf, show_confidence=True):
    """Render prediction results with beautiful styling"""
    is_dog = (pred == 0)
    
    # Result card
    if is_dog:
        st.markdown(f"""
            <div class="result-card">
                <p class="result-label">🐕 IT'S A DOG!</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="result-card result-card-negative">
                <p class="result-label">🚫 NOT A DOG!</p>                
            </div>
        """, unsafe_allow_html=True)
    
    # Additional info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{"✅" if is_dog else "❌"}</div>
                <div class="stat-label">Prediction</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{conf*100:.0f}%</div>
                <div class="stat-label">Confidence</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        certainty = "High" if conf > 0.9 else "Medium" if conf > 0.7 else "Low"
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{certainty}</div>
                <div class="stat-label">Certainty</div>
            </div>
        """, unsafe_allow_html=True)
