import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from mtcnn import MTCNN  # <-- Bunu ekledik

# --- AYARLAR ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "deepfake_resnet50.pth"

# --- MODELİ YÜKLE ---
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    # Eğer CPU kullanıyorsan map_location şart
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# --- YÜZ TESPİT MOTORU (MTCNN) ---
mtcnn = MTCNN()

# --- GÖRÜNTÜ İŞLEME (TRANSFORM) ---
# Burada artık Resize/CenterCrop yapmıyoruz çünkü MTCNN zaten yüzü kesti.
# Sadece modelin istediği boyuta (224x224) getiriyoruz.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("🕵️‍♂️ Deepfake Dedektörü v2.0")
st.write("Yapay Zeka Destekli (MTCNN + ResNet50)")

uploaded_file = st.file_uploader("Resim Yükle", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # Resmi PIL formatında aç
    image = Image.open(uploaded_file).convert('RGB')
    
    # Ekrana bas (Orijinal hali)
    st.image(image, caption="Yüklenen Resim", use_column_width=True)
    
    with st.spinner('Yüz aranıyor ve analiz ediliyor...'):
        # 1. ADIM: YÜZÜ BUL VE KES (Face Extraction)
        image_np = np.array(image) # MTCNN numpy sever
        faces = mtcnn.detect_faces(image_np)
        
        final_face = None
        
        if len(faces) > 0:
            # En büyük yüzü al (Birden fazla kişi varsa odaklanmak için)
            face = max(faces, key=lambda x: x['box'][2] * x['box'][3])
            x, y, w, h = face['box']
            
            # Koordinat düzeltmeleri (Negatif olmasın)
            x, y = max(0, x), max(0, y)
            
            # Yüzü biraz geniş al (Padding) - Model saç diplerini görsün
            padding = 0.2
            x_pad = int(w * padding)
            y_pad = int(h * padding)
            
            # Kırpma işlemi
            img_w, img_h = image.size
            x1 = max(0, x - x_pad)
            y1 = max(0, y - y_pad)
            x2 = min(img_w, x + w + x_pad)
            y2 = min(img_h, y + h + y_pad)
            
            final_face = image.crop((x1, y1, x2, y2))
            
            # Kullanıcıya neyi analiz ettiğimizi gösterelim
            st.image(final_face, caption="Tespit Edilen ve Analiz Edilen Yüz", width=200)
        
        else:
            st.warning("⚠️ Resimde net bir yüz bulunamadı! Tam resim analiz ediliyor.")
            final_face = image

        # 2. ADIM: MODELE SOR
        input_tensor = preprocess(final_face).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

        # 3. ADIM: SONUCU GÖSTER
        score = confidence.item() * 100
        label = "FAKE (SAHTE)" if predicted_class.item() == 0 else "REAL (GERÇEK)"
        
        # Renk ayarı
        color = "red" if predicted_class.item() == 0 else "green"

        st.markdown(f"## Sonuç: :{color}[{label}]")
        st.markdown(f"### Güven Skoru: %{score:.2f}")

        # Detaylı Çubuk
        st.write("Olasılık Dağılımı:")
        col1, col2 = st.columns(2)
        col1.metric("Fake Olasılığı", f"%{probs[0][0].item()*100:.2f}")
        col2.metric("Real Olasılığı", f"%{probs[0][1].item()*100:.2f}")