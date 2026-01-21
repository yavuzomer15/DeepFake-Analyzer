import data_exp
import data_preprocessing
import torchvision.models as models
import torch
from data_exp import device as device
from data_preprocessing import dataset_sizes
from data_preprocessing import dataLoaders
import copy

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

if device.type=="cuda":
    print(f"GPU model: {torch.cuda.get_device_name(0)}")

#epoch

def train_model(model, criterion, optimizer, num_epochs=3):
    best_model_wg=copy.deepcopy(model.state_dict())
    best_acc=0.0

    for epoch in range(num_epochs):
        print("Epoch starts..")

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss=0.0
            running_corrects=0

            for inputs,labels in dataLoaders[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)

                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs=model(inputs)
                    _, preds = torch.max(outputs,1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss+=loss.item()* inputs.size(0)
                running_corrects+=torch.sum(preds==labels.data)

            epoch_loss= running_loss / dataset_sizes[phase]
            epoch_acc=running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc=epoch_acc
                best_model_wg=copy.deepcopy(model.state_dict())
        print()
    model.load_state_dict(best_model_wg)        
    return model

model_ft=train_model(model,criterion,optimizer,num_epochs=3)
torch.save(model_ft.state_dict(),'deepfake_resnet50.pth')
print("Model has been saved!")
