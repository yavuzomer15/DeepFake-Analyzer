import data_exp
from torchvision import transforms
import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np

data_transform = {
    'train':transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test':transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
    ])
}

#Batching
BATCH_SIZE=32


train_data=torchvision.datasets.ImageFolder(root=data_exp.train_data_path, transform=data_transform['train'])
test_data=torchvision.datasets.ImageFolder(root=data_exp.test_data_path,transform=data_transform['test'])

image_datasets={
    'train': torchvision.datasets.ImageFolder(root=data_exp.train_data_path, transform=data_transform['train']),
    'test': torchvision.datasets.ImageFolder(root=data_exp.test_data_path,transform=data_transform['test'])
}


dataLoaders={
    'train': torch.utils.data.DataLoader(image_datasets['train'],batch_size=BATCH_SIZE,shuffle=True,num_workers=0),
    'test': torch.utils.data.DataLoader(image_datasets['test'],batch_size=BATCH_SIZE,shuffle=False,num_workers=0)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names= image_datasets['train'].classes

print(f"Dataset size: {dataset_sizes}")
print(f"Classes: {class_names}")



""""
# Görselleştirme fonksiyonu
def imshow(inp, title=None):
    #Tensoru resme çevirip gösterir
    inp = inp.numpy().transpose((1, 2, 0)) # (C, H, W) -> (H, W, C) formatına çevir
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean # Normalize işlemini geri al (Görüntü düzelsin)
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Bir batch (paket) veri çekelim
inputs, classes = next(iter(dataLoaders['train']))

# Grid (ızgara) yapalım
out = torchvision.utils.make_grid(inputs[:4]) # İlk 4 resmi al

plt.figure(figsize=(10, 5))
imshow(out, title=[class_names[x] for x in classes[:4]])
plt.show()"""