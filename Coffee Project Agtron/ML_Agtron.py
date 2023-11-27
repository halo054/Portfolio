import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

def read_label_csv(dataset_path):
    csv_path = dataset_path + '/label.csv'
    print(csv_path)
    # extract photo_name and Agtron Value from csv
    csv = pd.read_csv(csv_path)
    csv = csv.sample(frac = 1)
    
    Agtron_value = csv['Agtron']
    photo_name_list = csv['Photo']
    
    Agtron_value = np.array(Agtron_value)
    photo_name_list = np.array(photo_name_list)
    photo_name_list_train, photo_name_list_test, Agtron_value_train, Agtron_value_test = train_test_split(photo_name_list,Agtron_value, test_size=0.2, random_state=42)
    maximun = max(Agtron_value)
    minimun = min(Agtron_value)
    return photo_name_list_train, photo_name_list_test, Agtron_value_train, Agtron_value_test,maximun, minimun


class TrainDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.file_path = dataset_path
        photo_name_list,_,Agtron_value,_,maximun, minimun = read_label_csv(dataset_path)
        self.photo_name_list = photo_name_list
        self.Agtron_value = (Agtron_value - minimun) / (maximun - minimun)
        #self.Agtron_value = Agtron_value
        label_dict = {}
        i = 0
        for i in range(len(Agtron_value)):
            label_dict[photo_name_list[i]] = Agtron_value[i]
        self.label_dict = label_dict
        self.transform = transform
        
    
    def __len__(self):
        return len(self.photo_name_list)
    
    def __getitem__(self, idx):
        photo_name = self.photo_name_list[idx]
        Agtron_value = self.Agtron_value[idx]
        photo_path = self.file_path + '/' + photo_name
        photo = Image.open(photo_path)
        
        if self.transform:
            photo = self.transform(photo)
        
        return photo, Agtron_value

class TestDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.file_path = dataset_path
        _,photo_name_list,_,Agtron_value,maximun, minimun = read_label_csv(dataset_path)
        self.photo_name_list = photo_name_list
        self.Agtron_value = (Agtron_value  - minimun) / (maximun - minimun)
        #self.Agtron_value = Agtron_value
        label_dict = {}
        i = 0
        for i in range(len(Agtron_value)):
            label_dict[photo_name_list[i]] = Agtron_value[i]
        self.label_dict = label_dict
        self.transform = transform
        
    
    def __len__(self):
        return len(self.photo_name_list)
    
    def __getitem__(self, idx):
        photo_name = self.photo_name_list[idx]
        Agtron_value = self.Agtron_value[idx]
        photo_path = self.file_path + '/' + photo_name
        photo = Image.open(photo_path)
        
        if self.transform:
            photo = self.transform(photo)
        
        return photo, Agtron_value


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size= (1,1) , stride= 1,padding = 0, bias = True)
        self.conv2 = nn.Conv2d(20, 16, kernel_size= (1,1) , stride= 1,padding = 0, bias = True)
        self.conv3 = nn.Conv2d(16, 8, kernel_size= (3,3) , stride= 3,padding = 0, bias = True)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(3528, 1024, bias = True) 
        self.fc2 = nn.Linear(1024, 128, bias = True)
        self.fc3 = nn.Linear(128, 1, bias = True)
        
        
        self.avgpool = nn.AvgPool2d(2, stride = 2, padding=0)
        self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x):
        #print(x.size())
        x = self.conv1(x) 
        x = self.relu(x)
        x = self.avgpool(x)
       # print(x.size())
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
       # print(x.size())
       
        x = self.conv3(x)
        x = self.relu(x)
        x = self.avgpool(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu(x)
     #   print(x.size())
        
        x = self.fc2(x)
        x = self.relu(x)
       # print(x.size())
       
        x = self.fc3(x)
        return x


def train(epoch,model,loader,device,learning_rate = 1e-1):
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.2, weight_decay=1e-5)
    #MSE = 0
    L1 = 0
    for batch_idx, (photo, agtron) in enumerate(loader):
        
        photo = torch.tensor(photo, dtype=torch.double)
        photo = photo.requires_grad_(True).to(device)
        agtron = torch.tensor(agtron, dtype=torch.double)
        agtron = agtron.to(device)
        optimizer.zero_grad()
        pred_agtron = model(photo)
        loss = criterion(pred_agtron, agtron)
        L1 += loss.item()
        loss.backward()
        optimizer.step()
    print("epoch:",epoch+1,"L1 Loss:",L1/(batch_idx+1))
    print()

def test(model,loader,device,maximun,minimun):
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    #MSE = 0
    L1 = 0
    pred = []
    target = []
    for batch_idx, (photo, agtron) in enumerate(loader):
        photo = torch.tensor(photo, dtype=torch.double)
        photo = photo.to(device)
        agtron = torch.tensor(agtron, dtype=torch.double)
        agtron = agtron.to(device)
        agtron_pred = model(photo)
        loss = criterion(agtron_pred, agtron)
        L1 +=loss.item()
        
        
        agtron = agtron.detach().cpu().tolist()
        agtron_pred= agtron_pred.detach().cpu().tolist()
        
        
        target.extend(agtron)
        pred.extend(agtron_pred)
    pred = [value  for sublist in pred for value in sublist ]
    pred = np.array(pred)
    
    
    pred = pred * (maximun - minimun) + minimun
    
    target = np.array(target)  
    target = target * (maximun - minimun) + minimun
    print()
    print("pred:")
    print(pred)
    print()
    print("target:")
    print(target)
    print()
    
    
    print("test L1 = ",L1/(batch_idx+1))
        
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_path = "C:/Users/halo054/Desktop/Coffee Project Agtron/Dataset/Intact"
    batch_size = 10
    epoch = 1000
    learning_rate = 1e-2
    
    model = Model()
    model.to(device)
    model.to(torch.double)
    
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(512,scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])
    train_dataset = TrainDataset(dataset_path,transform = data_transform)
    test_dataset = TestDataset(dataset_path,transform = data_transform)
    trainloader = DataLoader(train_dataset, batch_size=batch_size)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    _,_,_,_,maximun,minimun = read_label_csv(dataset_path)
    #test(model,testloader,device)
    for i in range(epoch):
        train(i,model,trainloader,device,learning_rate)
    test(model,testloader,device,maximun,minimun)
main()