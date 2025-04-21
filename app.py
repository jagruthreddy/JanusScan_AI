import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

# Import custom modules
from modules.ai_detector import detect_ai_image
from modules.device_detector import detect_device
from modules.edit_detector import detect_edits, visualize_edits
from modules.metadata_analyzer import extract_metadata, recover_metadata
from modules.text_detector import detect_ai_text

# Set page configuration
st.set_page_config(
    page_title="JanusScan AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton button {
        background-color: #1E3A8A;
        color: white;
    }
    .result-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.title("JanusScan AI")
st.markdown("### Detect AI-generated images, smartphone sources, and image edits")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info(
        "JanusScan AI helps you verify the authenticity of images, "
        "detect AI-generated content, identify smartphone sources, "
        "and analyze edits, even for images that have lost metadata."
    )
    
    st.header("Features")
    st.markdown("""
    - AI Image Detection
    - Smartphone Source Identification
    - Edit Detection & Visualization
    - Metadata Recovery
    - AI Text Detection
    """)
    
    st.header("How it works")
    st.markdown("""
    Upload an image to analyze:
    1. We'll check if it's AI-generated
    2. Identify the camera source
    3. Detect and visualize edits
    4. Analyze any text for AI generation
    5. Recover metadata when possible
    """)

# Main app functionality
uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name
    
    # Load and display the image
    image = Image.open(temp_path)
    img_array = np.array(image)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Image Information")
        st.write(f"Format: {image.format}")
        st.write(f"Size: {image.size[0]} x {image.size[1]} pixels")
        st.write(f"Mode: {image.mode}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show analysis in progress
    with st.spinner("Analyzing image..."):
        # Extract metadata
        metadata_dict = extract_metadata(temp_path)
        
        # Check if it's an AI-generated image
        ai_result, ai_confidence = detect_ai_image(img_array)
        
        # Detect device
        device_result, device_confidence = detect_device(img_array, metadata_dict)
        
        # Detect edits
        is_edited, edit_map, edit_description = detect_edits(img_array)
        
        # Detect AI-generated text
        has_ai_text, text_confidence = detect_ai_text(img_array)
        
        # Attempt metadata recovery if needed
        recovered_metadata = None
        if not metadata_dict or len(metadata_dict) < 3:  # If minimal metadata found
            recovered_metadata = recover_metadata(img_array)
    
    # Display results
    st.markdown("## Analysis Results")
    
    # AI Detection Results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("AI Generation Detection")
        
        # Create gauge chart for AI confidence
        fig, ax = plt.subplots(figsize=(4, 0.3))
        ax.barh([0], [ai_confidence], color='#4CAF50' if not ai_result else '#F44336')
        ax.barh([0], [100], color='#E0E0E0', left=0, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.axis('off')
        
        st.pyplot(fig)
        
        if ai_result:
            st.error(f"✘ AI-generated image detected ({ai_confidence:.1f}% confidence)")
        else:
            st.success(f"✓ Natural image detected ({ai_confidence:.1f}% confidence)")
        
        if has_ai_text:
            st.warning(f"AI-generated text detected in image ({text_confidence:.1f}% confidence)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.subheader("Source Device")
        
        if device_result == "Unknown":
            st.info(f"Source device could not be determined with confidence")
        else:
            st.info(f"Likely captured with: **{device_result}** ({device_confidence:.1f}% confidence)")
        
        if recovered_metadata:
            st.success("✓ Partially recovered metadata suggests source device information")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Edit Detection
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader("Edit Detection")
    
    if is_edited:
        st.warning("✘ This image appears to have been edited")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(edit_map, caption="Edit Heatmap", use_column_width=True)
        with col2:
            st.markdown("### Detected Edits:")
            st.write(edit_description)
    else:
        st.success("✓ No significant edits detected")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Metadata Analysis
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.subheader("Metadata Analysis")
    
    if metadata_dict and len(metadata_dict) > 3:
        st.success("✓ Image contains metadata")
        
        with st.expander("View Metadata"):
            for key, value in metadata_dict.items():
                st.write(f"**{key}:** {value}")
    else:
        st.warning("⚠ Limited or no metadata found")
        if recovered_metadata:
            st.info("Some metadata was recovered using AI analysis")
            with st.expander("View Recovered Metadata"):
                for key, value in recovered_metadata.items():
                    st.write(f"**{key}:** {value}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clean up temporary file
    try:
        os.unlink(temp_path)
    except:
        pass
else:
    # Display demo/info when no image is uploaded
    st.info("Upload an image to begin analysis")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("""
        ### AI Detection
        Identifies images created by:
        - DALL-E
        - Midjourney
        - Stable Diffusion
        - Other AI models
        """)
    
    with col2:
        st.markdown("""
        ### Device Detection
        Identifies images from:
        - iPhone models
        - Samsung Galaxy
        - Google Pixel
        - Other smartphone cameras
        """)
    
    with col3:
        st.markdown("""
        ### Edit Analysis
        Detects and visualizes:
        - Photoshop edits
        - Object insertions
        - Color adjustments
        - Manipulations
        """)

# Footer
st.markdown("---")
st.markdown("© 2024 JanusScan AI | Developed for detecting AI-generated images and analyzing photos")
