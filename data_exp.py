import torch
import os
from pathlib import Path

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")

if device.type=="cuda":
    print(f"GPU model: {torch.cuda.get_device_name(0)}")

    print(f"VRAM:{torch.cuda.get_device_properties(0).total_memory / 1e9: .2f} GB")


data_path=Path("C:/Users/omery/Desktop/df/dataset")
train_data_path=Path("C:/Users/omery/Desktop/df/dataset/train")
test_data_path=Path("C:/Users/omery/Desktop/df/dataset/test")

def count_data(data_path):
    file_count=0
    for _,_,files in os.walk(data_path):
        file_count += len(files)
    return file_count

result_fake=count_data(data_path/"train/fake")
result_real=count_data(data_path/"train/real")
print(f"number of fake train data: {result_fake}")
print(f"number of real train data: {result_real}")