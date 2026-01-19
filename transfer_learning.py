import data_exp
import data_preprocessing
import torchvision.models as models
import torch
from data_exp import device as device

model= models.resnet50(weights='DEFAULT')

print(model)

for x in model.parameters():
    x.requires_grad=False


num_ftrs=model.fc.in_features
model.fc=torch.nn.Linear(num_ftrs,2)
model.to(device)

print("Model configuration operation is completed! Ready for model training!")

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.fc.parameters(),lr=0.001)

print("Loss function and optimizer arranged!")

#epoch

def train_model(model, inputs, labels, optimizer, criterion):
    
    optimizer.zero_grad()

    outputs = model(inputs)

    loss = criterion(outputs, labels)

    loss.backward()

    optimizer.step()