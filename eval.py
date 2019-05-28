import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
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

        print(f'[{self.usage}] Number of instances: {self.length} from Wiki dataset')

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
    
def get_data(batch_size):

    transform_test = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    data_test = ImdBDAgeDataset(usage='test', transform = transform_test)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)

    return test_loader

def test(data_loader, model):
    model.eval()   # Set model to evaluate mode
    sumCum = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f"[TEST]:"):
        data = data.to(device)
        target = target.float()
        target = target.to(device)

        with torch.set_grad_enabled(False):
            output = model(data)
            output = output.float()
            for i in range(output.shape[0]):
                sumCum += (abs(target[i]-output[i]))

    MAE = sumCum/len(data_loader.dataset)
    print(f"MAE = {MAE[0].detach().cpu().numpy():.4f} years")


if __name__=='__main__':
    # Batch size for training (change depending on how much memory you have)
    batch_size = 50

    test_loader = get_data(batch_size)

    model_ft = torch.load('Model_Age.pth')
    # Send the model to GPU
    model_ft = model_ft.to(device)
    test(test_loader,model_ft)


