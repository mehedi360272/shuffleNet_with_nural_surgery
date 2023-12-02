import numpy as np
import pandas as pd
import torch
from torch import optim
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
import os

# Create the Custom Dataset Class
from torch.utils.data import Dataset
class CustomDataset(Dataset):
  def __init__(self, X, y, BatchSize, transform):
    super().__init__()
    self.BatchSize = BatchSize
    self.y = y
    self.X = X
    self.transform = transform

  def num_of_batches(self):
    """
    Detect the total number of batches
    """
    return math.floor(len(self.list_IDs) / self.BatchSize)

  def __getitem__(self,idx):
    class_id = self.y[idx]
    img = self.X[idx].reshape(28,28)
    img = Image.fromarray(np.uint8(img * 255)).convert('L')
    img = self.transform(img)
    return img, torch.tensor(int(class_id))

  def __len__(self):
    return len(self.X)


# load data
df = pd.read_csv(r"/content/drive/MyDrive/Digital Image Processing/data_set/train.csv",dtype = np.float32)

# Shuffle dataframe
df = df.sample(frac=1)

# Split data into features X and labels y
X = df.loc[:, df.columns != "label"].values / 255
y = df.label.values

fig,ax = plt.subplots(2,5)
for i in range(10):
    nparray = X[i].reshape(28,28)
    image = Image.fromarray(nparray * 255)
    ax[i%2][i//2].imshow(image)
fig.show()



from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

# Define Transforms
transform = transforms.Compose([
                transforms.RandomRotation(10, fill=0),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                transforms.ToTensor(),
                transforms.RandomAffine(degrees=0, translate=(0.025, 0.025), fill=256),
                transforms.Normalize([0.5], [0.5])
            ])

test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

train_ratio = 0.90

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 - train_ratio, stratify = y, random_state = 0)

dataset_stages = ['train', 'val']

batch_size = 320
image_datasets = {'train' : CustomDataset(X_train, y_train, batch_size, transform), 'val' : CustomDataset(X_val, y_val, batch_size, test_transform)}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=image_datasets[x].BatchSize,
                                            shuffle=True, num_workers=0)
            for x in dataset_stages}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}




# Test Images from Dataset
fig,ax = plt.subplots(2,5)
for i in range(10):
    nparray = image_datasets['train'][i][0].cpu().numpy()
    image = transforms.ToPILImage()(image_datasets['train'][i][0].cpu()).convert("RGB")
    ax[i%2][i//2].imshow(image)
fig.show()





# Create a Training Function
import time
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            num_batches = 0
            outputs = None
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # Loading Bar
                if (phase == 'train'):
                    num_batches += 1
                    percentage_complete = ((num_batches * batch_size) / (dataset_sizes[phase])) * 100
                    percentage_complete = np.clip(percentage_complete, 0, 100)
                    print("{:0.2f}".format(percentage_complete), "% complete", end="\r")

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # TODO: try removal
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                predicted = torch.max(outputs.data, 1)[1]
                running_correct = (predicted == labels).sum()
                running_corrects += running_correct
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects / dataset_sizes[phase]
            #epoch_acc = sum(epoch_acc) / len(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc.item()))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model




# Load up Shufflenet

from torchvision import models
from torch.optim import lr_scheduler

shufflenet = models.shufflenet_v2_x1_0()
shufflenet.conv1[0] = nn.Conv2d(1, 24, kernel_size=(2, 2), stride=(1, 1))
shufflenet.fc = nn.Linear(in_features=1024, out_features=10, bias=True)
model_ft = shufflenet



criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

shufflenet = train_model(shufflenet.to(device), criterion, optimizer_ft, exp_lr_scheduler, 35)
