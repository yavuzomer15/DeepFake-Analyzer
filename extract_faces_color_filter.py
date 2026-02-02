import cv2
import os
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
import logging
import tensorflow as tf

# TensorFlow uyarılarını kapa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

# --- AYARLAR ---
SOURCE_DIR = "FACES2"
DEST_DIR = "FACES3"
PADDING = 0.20
CONFIDENCE_THRESHOLD = 0.95

# 🎨 RENK EŞİĞİ (KRİTİK AYAR)
# 0 = Tamamen Gri, 255 = Çok Canlı Renkler
# 20 değeri genelde siyah-beyaz ve çok soluk resimleri elemek için idealdir.
SATURATION_THRESHOLD = 20 

def is_color_image(image_bgr):
    """
    Resmin renkli olup olmadığını kontrol eder.
    HSV formatına çevirip 'S' (Saturation/Doygunluk) kanalının ortalamasına bakar.
    """
    # HSV formatına çevir (Hue, Saturation, Value)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Sadece Saturation (Doygunluk) kanalını al
    saturation = hsv[:, :, 1]
    
    # Ortalamasını hesapla
    mean_sat = np.mean(saturation)
    
    # Eğer ortalama doygunluk eşiğin altındaysa, bu resim gri/siyah-beyazdır.
    if mean_sat < SATURATION_THRESHOLD:
        return False # Renkli Değil
    return True # Renkli

def extract_faces_clean():
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    print("🧠 MTCNN Modeli ve Renk Filtresi Yükleniyor...")
    detector = MTCNN()

    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    print(f"📂 Toplam Dosya: {len(files)}")
    
    saved_count = 0
    skipped_bw = 0      # Atılan Siyah Beyazlar
    skipped_low_conf = 0 # Atılan Düşük Güvenli Yüzler

    for filename in tqdm(files, desc="🚀 Tarama (B&W Filtreli)"):
        img_path = os.path.join(SOURCE_DIR, filename)
        
        try:
            image = cv2.imread(img_path)
            if image is None: continue

            # --- 1. SİYAH BEYAZ KONTROLÜ (İşlemden önce yapıyoruz ki boşuna vakit harcamasın) ---
            if not is_color_image(image):
                skipped_bw += 1
                continue # Döngünün başına dön, bu resmi atla

            # --- 2. Yüz Tespiti (MTCNN) ---
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(image_rgb)

            h_img, w_img, _ = image.shape

            for result in results:
                if result['confidence'] < CONFIDENCE_THRESHOLD:
                    skipped_low_conf += 1
                    continue

                x, y, w, h = result['box']
                x, y = max(0, x), max(0, y)

                # Padding
                x_pad = int(w * PADDING)
                y_pad = int(h * PADDING)

                x_start = max(0, x - x_pad)
                y_start = max(0, y - y_pad)
                x_end = min(w_img, x + w + x_pad)
                y_end = min(h_img, y + h + y_pad)

                face_crop = image[y_start:y_end, x_start:x_end]

                if face_crop.size > 0:
                    save_name = f"face_{saved_count}_{filename}".replace(" ", "_")
                    save_name = os.path.splitext(save_name)[0] + ".jpg"
                    
                    save_path = os.path.join(DEST_DIR, save_name)
                    cv2.imwrite(save_path, face_crop)
                    saved_count += 1

        except Exception as e:
            continue

    print("-" * 30)
    print(f"✅ İŞLEM TAMAMLANDI!")
    print(f"🎉 Kaydedilen Renkli Yüz: {saved_count}")
    print(f"⚫ Atılan Siyah/Beyaz Resim: {skipped_bw}")
    print(f"🗑️ Güvenilir Bulunmayan Yüz: {skipped_low_conf}")

if __name__ == "__main__":
    extract_faces_clean()