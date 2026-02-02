import cv2
import os
from mtcnn import MTCNN
from tqdm import tqdm
import logging
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

SOURCE_DIR = "C:/Users/omery/Desktop/df/male_female"
DEST_DIR = "FACES2"
PADDING = 0.20  
CONFIDENCE_THRESHOLD = 0.95 

def extract_faces_mtcnn():

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    print("MTCNN Model Uploading.. Please Wait")
    detector = MTCNN()

    files = [f for f in os.listdir(SOURCE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    print(f"📂 Total Files: {len(files)}")
    
    saved_count = 0
    skipped_count = 0

    for filename in tqdm(files, desc="Scanning Faces"):
        img_path = os.path.join(SOURCE_DIR, filename)
        
        try:
            # 1. Resmi OpenCV ile Oku
            image = cv2.imread(img_path)
            if image is None: continue

            # 2. BGR -> RGB Çevir (MTCNN RGB sever)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 3. Yüzleri Bul
            results = detector.detect_faces(image_rgb)

            h_img, w_img, _ = image.shape

            for result in results:
                # Güven kontrolü
                if result['confidence'] < CONFIDENCE_THRESHOLD:
                    continue

                # Koordinatları al
                x, y, w, h = result['box']
                # Negatif değerleri düzelt
                x, y = max(0, x), max(0, y)

                # --- PADDING ---
                x_pad = int(w * PADDING)
                y_pad = int(h * PADDING)

                x_start = max(0, x - x_pad)
                y_start = max(0, y - y_pad)
                x_end = min(w_img, x + w + x_pad)
                y_end = min(h_img, y + h + y_pad)

                # Kırp (Orijinal BGR resimden)
                face_crop = image[y_start:y_end, x_start:x_end]

                if face_crop.size > 0:
                    save_name = f"face_{saved_count}_{filename}".replace(" ", "_")
                    # Uzantıyı .jpg yapalım (yer kaplamasın)
                    save_name = os.path.splitext(save_name)[0] + ".jpg"
                    
                    save_path = os.path.join(DEST_DIR, save_name)
                    cv2.imwrite(save_path, face_crop)
                    saved_count += 1
                else:
                    skipped_count += 1

        except Exception as e:
            # Hata olursa basma, devam et
            continue

    print("-" * 30)
    print(f"✅ İŞLEM TAMAMLANDI!")
    print(f"🎉 Kaydedilen Temiz Yüz: {saved_count}")
    print(f"🗑️ Atılan / Bulunamayan: {len(files) - saved_count}")

if __name__ == "__main__":
    extract_faces_mtcnn()