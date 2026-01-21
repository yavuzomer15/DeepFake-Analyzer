import os
from PIL import Image
import warnings

warnings.filterwarnings("error")

dataset_path="dataset"

def check_image(root_dir):
    print(f"Cleaning starts: {root_dir}")
    bad_files=0
    checked_files=0

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            filepath=os.path.join(subdir,file)

            if file.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff')):
                checked_files+=1
                try:
                    with Image.open(filepath) as img2:
                        img2.load()
                except(IOError,SyntaxError,OSError) as e:
                    print(f"Bad files find and removed: {filepath}")
                    print(f"Reason: {e}")
                    try:
                        os.remove(filepath)
                        bad_files+=1
                    except:
                        print("Could not be removed! Please remove manually!")
    print("Finished.")
    print(f"Total checked: {checked_files}")
    print(f"Total removed: {bad_files}")

if __name__ == "__main__":
    if os.path.exists(dataset_path):
        check_image(dataset_path)
    else:
        print(f"Error: {dataset_path} could not find.")
