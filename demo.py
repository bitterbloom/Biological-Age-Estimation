import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from random import randint
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_mat(path):
    from scipy.io import loadmat
    mat = loadmat(path)
    return mat["paths"], mat["gender"][0], mat["age"][0]

class ImdBDAgeDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, usage, transform):
        'Initialization'
        self.usage = usage
        self.transform = transform
        mat_path = './data/wiki_db.mat'
        self.paths,self.gender,self.age = load_mat(mat_path)

        test = {
            "path":self.paths,
            "age":self.age,
            "gender":self.gender
        }

        self.data = test
        self.length = len(self.paths)

  def __len__(self):
        'Denotes the total number of samples'
        return self.length

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        path = self.data["path"][index].split(" ")[0]
        age = self.data["age"][index]
        # Load data and get label
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        label = age

        return img, label
    
def get_data():

    transform_test = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    data_test = ImdBDAgeDataset(usage='test', transform = transform_test)

    return data_test

if __name__=='__main__':
    #Load data and model
    test_loader = get_data()
    model_ft = torch.load('Age_lr_0.01_resnet.pth')

    # Send the model to GPU 
    model_ft = model_ft.to(device)
    model_ft.eval()

    #Random image
    rng = randint(0,len(test_loader))
    image,label = test_loader[rng]
    img = image.view(1,3,224,224)

    #Send image to GPU
    img = img.to(device)

    #Calculate age
    with torch.set_grad_enabled(False):
        out = model_ft(img)
    
    #Plot
    image = image.permute(1,2,0).numpy()
    pred = out.detach().cpu().numpy()
    pred = pred[0][0]
    plt.imshow(image)
    plt.title(f'Image #{rng} with Label = {label:.2f} and Predicted = {pred:.2f}')
    plt.savefig("demo.png")
    PilIm = Image.open("demo.png")
    PilIm.convert("RGB")
    PilIm.save("demo.png")
    PilIm.show()


