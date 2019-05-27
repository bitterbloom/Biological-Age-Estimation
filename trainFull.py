import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
import torch.utils.model_zoo as model_zoo
from torch.nn import _reduction as _Reduction
import numpy as np
import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_mat(path):
    from scipy.io import loadmat
    mat = loadmat(path)
    return mat["paths"], mat["gender"][0], mat["age"][0]


def AgeProb(input):
    mat_path = './data/imdb_db.mat'
    _,_,age = load_mat(mat_path)
    cats = np.unique(age)

    bins = np.zeros(cats.shape[0])
    for i in cats:
        count = np.count_nonzero(age == i)
        bins[i] = count/age.shape[0]
    
    probs = []
    for i in input:
        probs.append(bins[i])
    
    return torch.tensor(probs)

class ImdBDAgeDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, usage, transform):
        'Initialization'
        self.usage = usage
        self.transform = transform
        mat_path = './data/imdb_db.mat'
        self.paths,self.gender,self.age = load_mat(mat_path)
        
        trainP,valP = train_test_split(self.paths,test_size=0.2,random_state=0)
        trainG,valG = train_test_split(self.gender,test_size=0.2,random_state=0)
        trainA,valA = train_test_split(self.age,test_size=0.2,random_state=0)
        
        if self.usage == 'train':
            train ={
            "path":trainP,
            "age":trainA,
            "gender":trainG
            }
            self.data =  train
            self.length = len(trainP)
        elif self.usage == 'val':
            val ={
            "path":valP,
            "age":valA,
            "gender":valG
            }
            self.data = val
            self.length = len(valP)
        # elif self.usage == 'test':
        #     test = os.listdir('./data/wiki_crop')
        #     self.files_path = test

        print(f'[{self.usage}] Number of instances: {self.length}')

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
    transform_train = transforms.Compose(
        [transforms.Resize((256,256)),
        transforms.RandomCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    transform_test = transforms.Compose(
        [transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    data_train = ImdBDAgeDataset(usage='train', transform = transform_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_val = ImdBDAgeDataset(usage='val', transform = transform_test)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, data_val

def train(data_loaders, model, optimizer, epoch):
    best_model_wts = copy.deepcopy(model.state_dict())
    # Each epoch has a training and validation phase
    for phase in ['train']:
        if phase == 'train':
            model.train()  # Set model to training mode

        loss_cum = []
        Acc = 0
        for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loaders[phase]), total=len(data_loaders[phase]), desc=f"[{phase}] Epoch: {epoch}"):
            data = data.to(device)
            target = target.float()
            target = target.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                output = model(data)
                output = output.float()

                loss = mse_loss(output.flatten(),target)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                loss_cum.append(loss.item())

            _, arg_max_out = torch.max(output.data.cpu(), 1)
            Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()

        print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loaders[phase].dataset)))

        if phase == 'train':
            train_loss = np.array(loss_cum).mean()
            

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet101
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes,bias=False)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes,bias=False)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :class:`~torch.nn.MSELoss` for details.
    """
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = ((1-AgeProb(input))^20)*(input - target) ** 2
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret

def eval_train(data_loader, model, optimizer, epoch):
    model.eval()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc=f"[Train Eval] Epoch: {epoch}"):
        data = data.to(device)
        target = target.float()
        target = target.to(device)
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            output = model(data)
            output = output.float()

            loss = mse_loss(output.flatten(),target)
            loss_cum.append(loss.item())

        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()

    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))
    return np.array(loss_cum).mean()

if __name__=='__main__':
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    for model_name in ["resnet"]:

        # Number of classes in the dataset
        num_classes = 1

        # Batch size for training (change depending on how much memory you have)
        batch_size = 25

        # Number of epochs to train for
        num_epochs = 15

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = False

        train_loader, val_loader, data_val = get_data(batch_size)
        training_loaders = {
            "train": train_loader,
            #"val": val_loader 
        }

    # Initialize the model for this run
        #path = 'g20_lr0.01_E25_resnet_3.pth'
        #print(path)
        #model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        model_ft = torch.load(path)
        # Send the model to GPU
        model_ft = model_ft.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(params_to_update,lr=0.01)

        # Train and evaluate
        f = open(f"g20_lr0.01_E15_{model_name}.txt","w+")
        for epoch in range(num_epochs):
            model_ft, train_loss = train(training_loaders,model_ft,optimizer_ft,epoch)
            f.write(f"Epoch: {epoch}, train_loss:{train_loss}\n")
            torch.save(model_ft,f"g20_lr0.01_E15_{model_name}_{epoch}.pth")
        f.close()
        torch.save(model_ft,f"Model_Age.pth")
