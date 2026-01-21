import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import os
from PIL import Image

model_path="C:/Users/omery/Desktop/df/deepfake_resnet50.pth"
image_path="C:/Users/omery/Desktop/df/test_image.jpeg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(path):

    print("Model is arranging..")

    model=models.resnet50(weights=None)

    model.fc=nn.Linear(2048, 2)

    model_load=model.load_state_dict(torch.load(path))

    model.to(device)

    model.eval()

    return model

def image_pre(model, image_path):
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except:
        print("Image could not be opened")  
        
    image_tensor=transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs=model(image_tensor)

        prob=nn.functional.softmax(outputs, dim=1)[0]*100
        _, pred=torch.max(outputs,1)

        class_name=['fakes','real']
        print(f"Result:{class_name[pred[0]]}")
        print(f"Confidence:{prob[pred[0]].item():.2f}")

model =load_model(model_path)
image_pre(model, image_path)