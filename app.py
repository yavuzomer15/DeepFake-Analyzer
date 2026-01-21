import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = "deepfake_resnet50.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def local_css():
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #FF4B4B;
            text-align: center;
            font-weight: 700;
            margin-bottom: 0px;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #808495;
            text-align: center;
            margin-bottom: 20px;
        }
        .upload-area {
            border: 2px dashed #4B4B4B;
            padding: 20px;
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 2)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        return None


def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),       
        transforms.CenterCrop(224),   
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


def main():
    st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️‍♂️", layout="wide")
    local_css() 

    
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/9283/9283308.png", width=80)
        st.title("About Project")
        st.info(
            """
            This system uses a **ResNet50** Convolutional Neural Network trained to distinguish between:
            
            * **Real Faces** (FFHQ / CelebA)
            * **AI Generated Faces** (StyleGAN / Midjourney)
            """
        )
        st.write("---")
        st.caption(f"🚀 Device: **{DEVICE}**")
        st.caption("👨‍💻 Developer: Ömer Yavuz")

   
    st.markdown('<p class="main-header">🕵️‍♂️ Deepfake Analysis System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Forensic Tool for Synthetic Image Detection</p>', unsafe_allow_html=True)
    
    st.divider() 

    
    with st.spinner("🚀 Booting up the neural network..."):
        model = load_model()

    if model is None:
        st.error(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        return

    
    col_upload, col_empty = st.columns([8, 2]) 
    
    with col_upload:
        st.subheader("1. Upload Image")
        st.markdown("Please upload a high-quality image of a face.")
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, PNG")

    
    if uploaded_file is not None:
        st.divider() 
        
        col1, col2 = st.columns([1, 1])
        
        
        with col1:
            st.subheader("2. Input Image")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, use_container_width=True, caption="Source Image")

        
        with col2:
            st.subheader("3. Analysis Results")
            
            
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Processing pixels..."):
                    tensor = process_image(image)
                    
                    with torch.no_grad():
                        outputs = model(tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                        conf, pred = torch.max(probs, 0)
                        
                        
                        class_names = ['FAKE / AI GENERATED', 'REAL / AUTHENTIC']
                        
                        result_text = class_names[pred]
                        confidence_score = conf.item()
                        
                        
                        
                        
                        if pred == 0: 
                            st.error(f"🚨 DETECTION: **{result_text}**")
                        else: 
                             
                            if confidence_score < 75.0:
                                st.warning(f"⚠️ RESULT: **SUSPICIOUS / UNCERTAIN**")
                                st.caption("The model identified this as Real, but the confidence is low. It might be an artistic drawing or highly edited photo.")
                            else:
                                st.success(f"✅ RESULT: **{result_text}**")

                        
                        st.write("") 
                        col_metric1, col_metric2 = st.columns(2)
                        with col_metric1:
                             st.metric(label="Confidence Score", value=f"%{confidence_score:.2f}")
                        
                        
                        st.write("Probability Distribution:")
                        st.progress(int(probs[1].item())) 
                        
                        with st.expander("See Detailed Probabilities"):
                            st.write(f"🤖 Artificial/Fake: **%{probs[0]:.2f}**")
                            st.write(f"👤 Real/Human: **%{probs[1]:.2f}**")

if __name__ == "__main__":
    main()