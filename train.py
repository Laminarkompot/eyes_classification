import numpy as np
from torchvision import models, datasets, transforms
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


TRAIN_DATA_PATH = '/home/den/code/edu/MynaLabs/train/'
TEST_DATA_PATH = '/home/den/code/edu/MynaLabs/test/'

BATCH_SIZE = 180
DEVICE = 'cuda'
NUM_EPOCHS = 25


transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.Resize(64)
])

train_data = datasets.ImageFolder(
    root=TRAIN_DATA_PATH, transform=transforms
)

test_data = datasets.ImageFolder(
    root=TEST_DATA_PATH, transform=transforms
)

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)


model = models.resnet18(pretrained=True)

for name, param in model.named_parameters():
    if ('bn' not in name):
        param.requires_grad = False

classifier = nn.Sequential(nn.Dropout(0.5),
                           nn.Linear(512,2))

model.fc = classifier
# model.fc.requires_grad = True
print(model)
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

test_acc_array = []
epochs = []

for epoch in range(NUM_EPOCHS):
    print('Epoch', epoch)
    model.train()
    for batch_X, batch_y in tqdm(train_data_loader):
        # Get batch of data
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        # Pass to the neural network
        out = model(batch_X) # [B, 10]
        # Compute loss
        loss = loss_fn(out, batch_y)
        # Gradient descent step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # Predict on validation set
    outputs = []
    y_val = []
    model.eval()
    for batch_X, batch_y in test_data_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        out = model(batch_X) # [21000, 10]
        out = out.detach().cpu().numpy()
        out = np.argmax(out, axis=1)
        outputs.append(out)
        y_val.append(batch_y.cpu().numpy())
    outputs = np.concatenate(outputs)
    y_val = np.concatenate(y_val)
    # Compute accuracy
    acc = (outputs == y_val).mean()
    test_acc_array.append(acc * 100)
    print('ACCURACY    ', acc)
    epochs.append(epoch + 1)
    

    # Define figure for plotting
    fig = plt.figure(figsize=(20, 10))
    # Acc vs number of epochs
    x_hor = [0, epochs[-1]]
    y_100 = [100, 100]
    # epochs = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs, test_acc_array, label='test_acc')
    plt.plot(x_hor, y_100, linestyle='--', color='grey', label='100%')
    plt.axis([0, NUM_EPOCHS, 0, 110])
    plt.legend()
    plt.xlabel('Количество эпох')
    plt.ylabel('Точность, %')

    # Save figure
    fig.savefig('plot.png')

torch.save(model, 'ResNet18_eyes')
torch.save(model.state_dict(), 'ResNet18_eyes_dict')