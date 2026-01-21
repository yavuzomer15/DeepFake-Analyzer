import os
import shutil
import random
from tqdm import tqdm

real_dir="Temp_Pool/All_Reals"
fake_dir="Temp_Pool/All_Fakes"

dataset_dir="dataset"

split_ratio=0.8

def create_dir_struct():
    paths = [
        (os.path.join(dataset_dir,'train','real')),
        (os.path.join(dataset_dir,'train','fake')),
        (os.path.join(dataset_dir,'test','real')),
        (os.path.join(dataset_dir,'test','fake'))
    ]
    for path in paths:
        os.makedirs(path,exist_ok=True)
    print("File structure created")

def split_and_copy(source_dir,class_name):
    print(f"{class_name} data is processing..")
    
    files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files)

    split_point=int(len(files)*split_ratio)
    train_files=files[:split_point]
    test_files=files[split_point:]

    print(f"Total: {len(files)}")
    print(f"To train:{len(train_files)}")
    print(f"To Test:{len(test_files)}")

    def copy_files(file_list,split_type):
        dest_dir=os.path.join(dataset_dir,split_type,class_name)
        for filename in tqdm(file_list, desc=f"{split_type} copying..."):
            src=os.path.join(source_dir,filename)
            dst=os.path.join(dest_dir,filename)
            shutil.copy2(src,dst)

    copy_files(train_files,'train')
    copy_files(test_files,'test')


if __name__ == "__main__":
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("File can not be find.")
    else:
        create_dir_struct()
        split_and_copy(real_dir, 'real')
        split_and_copy(fake_dir, 'fake')
        print("Completed!")