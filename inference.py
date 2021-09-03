import numpy as np
import pandas as pd
import torch
import os
import nnio
from sys import argv

# Define constants
NET_PATH = '/home/den/code/edu/MynaLabs/'
SCRIPTNAME, DATA_PATH = argv
DEVICE = 'cuda'

# Load model
model = torch.load(NET_PATH + 'ResNet18_eyes')

# Pass model to CUDA
model.to(DEVICE)

# Get file names
file_names = sorted(os.listdir(DATA_PATH))

# Define data preprocessing
preproc = nnio.Preprocessing(
    resize=(64, 64),
    dtype='float32',
    divide_by_255=True,
    means=[0.485, 0.456, 0.406],
    stds=[0.229, 0.224, 0.225],
    batch_dimension=True,
    channels_first=True,
)

# Define array for predictions
outputs = []

# Start classification
model.eval()
for i in range(len(file_names)):
    img = preproc(DATA_PATH + file_names[i])
    img = torch.tensor(img, dtype=torch.float32, device=DEVICE)
    out = model(img)
    out = out.detach().cpu().numpy()
    out = np.argmax(out, axis=1)[0]
    outputs.append(out)
    file_names[i] = DATA_PATH + file_names[i]
    print(file_names[i], '  predicted label: ', out)

print(len(file_names), 'files processed. Writing to CSV...')

# Write results to CSV
results = pd.Series(outputs, index=file_names)
results.to_csv('results.csv', header=False)
print('Done.')