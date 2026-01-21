import torch
from torchvision import datasets
import os

# Veri setinin olduğu klasör (Train klasörü)
data_dir = "C:/Users/omery/Desktop/df/dataset/real-vs-fake/train" # Kendi yolunu kontrol et!

if os.path.exists(data_dir):
    dataset = datasets.ImageFolder(data_dir)
    print("\n--- GERÇEK ETİKET SIRALAMASI ---")
    print(dataset.class_to_idx)
    print("--------------------------------\n")
    
    # İpucu
    mapping = dataset.class_to_idx
    if mapping.get('fake') == 0:
        print("Modelin dili: 0 -> FAKE, 1 -> REAL")
        print("predict.py içindeki listen şöyle olmalı: class_names = ['fake', 'real']")
    else:
        print("Modelin dili: 0 -> REAL, 1 -> FAKE")
        print("predict.py içindeki listen şöyle olmalı: class_names = ['real', 'fake']")
else:
    print("Hata: Dataset klasörü bulunamadı. Lütfen yolu düzelt.")