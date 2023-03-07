# -*- coding: utf-8 -*-
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
from frac_init import yuanhan_normal_, randomwalk_normal_


# To use different initialization method, one can decomment the code
 
class simpleNet(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()

        # 20 layer-Fully connected network with the new initialization method
        # We choose the s = 1 for Leaky-ReLU activation

        
        self.layer1 = nn.Linear(in_dim, n_hidden_2)
        yuanhan_normal_(self.layer1.weight,nonlinearity='leaky_relu', s=1)
        self.layer2 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer2.weight,nonlinearity='leaky_relu', s=1)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer3.weight,nonlinearity='leaky_relu', s=1)
        self.layer4 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer4.weight,nonlinearity='leaky_relu', s=1)
        self.layer5 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer5.weight,nonlinearity='leaky_relu', s=1)
        self.layer6 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer6.weight,nonlinearity='leaky_relu', s=1)
        self.layer7 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer7.weight,nonlinearity='leaky_relu', s=1)
        self.layer8 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer8.weight,nonlinearity='leaky_relu', s=1)
        self.layer9 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer9.weight,nonlinearity='leaky_relu', s=1)
        self.layer10 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer10.weight,nonlinearity='leaky_relu', s=1)
        self.layer11 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer11.weight,nonlinearity='leaky_relu', s=1)
        self.layer12 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer12.weight,nonlinearity='leaky_relu', s=1)
        self.layer13 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer13.weight,nonlinearity='leaky_relu', s=1)
        self.layer14 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer14.weight,nonlinearity='leaky_relu', s=1)
        self.layer15 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer15.weight,nonlinearity='leaky_relu', s=1)
        self.layer16 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer16.weight,nonlinearity='leaky_relu', s=1)
        self.layer17 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer17.weight,nonlinearity='leaky_relu', s=1)
        self.layer18 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer18.weight,nonlinearity='leaky_relu', s=1)
        self.layer19 = nn.Linear(n_hidden_2, n_hidden_2)
        yuanhan_normal_(self.layer19.weight,nonlinearity='leaky_relu', s=1)
        self.layer20 = nn.Linear(n_hidden_2, out_dim)
        yuanhan_normal_(self.layer20.weight,nonlinearity='leaky_relu', s=1)
        
        
        
        # Kaiming initialization method
        
        '''
        self.layer1 = nn.Linear(in_dim, n_hidden_2)
        nn.init.kaiming_normal_(self.layer1.weight,nonlinearity='leaky_relu')
        self.layer2 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer2.weight,nonlinearity='leaky_relu')
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer3.weight,nonlinearity='leaky_relu')
        self.layer4 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer4.weight,nonlinearity='leaky_relu')
        self.layer5 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer5.weight,nonlinearity='leaky_relu')
        self.layer6 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer6.weight,nonlinearity='leaky_relu')
        self.layer7 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer7.weight,nonlinearity='leaky_relu')
        self.layer8 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer8.weight,nonlinearity='leaky_relu')
        self.layer9 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer9.weight,nonlinearity='leaky_relu')
        self.layer10 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer10.weight,nonlinearity='leaky_relu')
        self.layer11 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer11.weight,nonlinearity='leaky_relu')
        self.layer12 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer12.weight,nonlinearity='leaky_relu')
        self.layer13 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer13.weight,nonlinearity='leaky_relu')
        self.layer14 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer14.weight,nonlinearity='leaky_relu')
        self.layer15 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer15.weight,nonlinearity='leaky_relu')
        self.layer16 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer16.weight,nonlinearity='leaky_relu')
        self.layer17 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer17.weight,nonlinearity='leaky_relu')
        self.layer18 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer18.weight,nonlinearity='leaky_relu')
        self.layer19 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.kaiming_normal_(self.layer19.weight,nonlinearity='leaky_relu')
        self.layer20 = nn.Linear(n_hidden_2, out_dim)
        nn.init.kaiming_normal_(self.layer20.weight,nonlinearity='leaky_relu')
        '''
        
        # Xavier initialization method
        
        '''
        self.layer1 = nn.Linear(in_dim, n_hidden_2)
        nn.init.xavier_normal_(self.layer1.weight)
        self.layer2 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer2.weight)
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer3.weight)
        self.layer4 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer4.weight)
        self.layer5 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer5.weight)
        self.layer6 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer6.weight)
        self.layer7 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer7.weight)
        self.layer8 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer8.weight)
        self.layer9 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer9.weight)
        self.layer10 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer10.weight)
        self.layer11 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer11.weight)
        self.layer12 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer12.weight)
        self.layer13 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer13.weight)
        self.layer14 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer14.weight)
        self.layer15 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer15.weight)
        self.layer16 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer16.weight)
        self.layer17 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer17.weight)
        self.layer18 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer18.weight)
        self.layer19 = nn.Linear(n_hidden_2, n_hidden_2)
        nn.init.xavier_normal_(self.layer19.weight)
        self.layer20 = nn.Linear(n_hidden_2, out_dim)
        nn.init.xavier_normal_(self.layer20.weight)
        '''
        
        # random walk initialization method
        
        '''
        self.layer1 = nn.Linear(in_dim, n_hidden_2)
        randomwalk_normal_(self.layer1.weight,nonlinearity='relu')
        self.layer2 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer2.weight,nonlinearity='relu')
        self.layer3 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer3.weight,nonlinearity='relu')
        self.layer4 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer4.weight,nonlinearity='relu')
        self.layer5 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer5.weight,nonlinearity='relu')
        self.layer6 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer6.weight,nonlinearity='relu')
        self.layer7 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer7.weight,nonlinearity='relu')
        self.layer8 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer8.weight,nonlinearity='relu')
        self.layer9 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer9.weight,nonlinearity='relu')
        self.layer10 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer10.weight,nonlinearity='relu')
        self.layer11 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer11.weight,nonlinearity='relu')
        self.layer12 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer12.weight,nonlinearity='relu')
        self.layer13 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer13.weight,nonlinearity='relu')
        self.layer14 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer14.weight,nonlinearity='relu')
        self.layer15 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer15.weight,nonlinearity='relu')
        self.layer16 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer16.weight,nonlinearity='relu')
        self.layer17 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer17.weight,nonlinearity='relu')
        self.layer18 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer18.weight,nonlinearity='relu')
        self.layer19 = nn.Linear(n_hidden_2, n_hidden_2)
        randomwalk_normal_(self.layer19.weight,nonlinearity='relu')
        self.layer20 = nn.Linear(n_hidden_2, out_dim)
        randomwalk_normal_(self.layer20.weight,nonlinearity='relu')
        '''

        
        
    def forward(self, x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        x = F.leaky_relu(self.layer4(x))
        x = F.leaky_relu(self.layer5(x))
        x = F.leaky_relu(self.layer6(x))
        x = F.leaky_relu(self.layer7(x))
        x = F.leaky_relu(self.layer8(x))
        x = F.leaky_relu(self.layer9(x))
        x = F.leaky_relu(self.layer10(x))
        x = F.leaky_relu(self.layer11(x))
        x = F.leaky_relu(self.layer12(x))
        x = F.leaky_relu(self.layer13(x))
        x = F.leaky_relu(self.layer14(x))
        x = F.leaky_relu(self.layer15(x))
        x = F.leaky_relu(self.layer16(x))
        x = F.leaky_relu(self.layer17(x))
        x = F.leaky_relu(self.layer18(x))
        x = F.leaky_relu(self.layer19(x))
        x = F.leaky_relu(self.layer20(x))
        return x


# load MNIST data
batch_size = 128
lossList=[]
lossList2=[]
accList=[]
accList2=[]
data_tf = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(
        root='./data', train=True, transform=data_tf, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_tf)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# We train 20 different networks to calculate the averaged accuracy and loss
for time in range(20):
    
    batch_size = 128
    learning_rate = 0.01
    # num_epoches = 50
    
    # initialize the model
    model = simpleNet(28 * 28, 64, 64, 10)

    if torch.cuda.is_available():
        model = model.cuda()
     

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    epoch = 30
    trainErrorList=[]
    testErrorList=[]
    trainAccList=[]
    testAccList=[]
    #print(train_loader)
    for i in range(epoch):
        train_acc=0
        test_acc=0
        # update the model and calculate the training accuracy
        for data in train_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            loss = criterion(out, label)
            print_loss = loss.data.item()
            _, pred = torch.max(out.data, 1)
            train_acc += pred.eq(label.view_as(pred)).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        trainErrorList.append(loss.data.item())
        trainAccList.append(train_acc/60000)
        print('epoch: {},train loss: {:.4}'.format(i, loss.data.item()))
        
        # calculate the test accuracy
        for data in test_loader:
            img, label = data
            img = img.view(img.size(0), -1)
            if torch.cuda.is_available():
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            out = model(img)
            loss = criterion(out, label)
            print_loss = loss.data.item()
            #accuracy=F.accuracy(out,label)
            _, pred = torch.max(out.data, 1)
            test_acc += pred.eq(label.view_as(pred)).sum().item()

        testErrorList.append(loss.data.item())
        testAccList.append(test_acc/10000)
        
    lossList.append(trainErrorList)
    lossList2.append(testErrorList)
    accList.append(trainAccList)
    accList2.append(testAccList)

# Calculate the averaged accuracy and loss
avg=[]
avg2=[]
avgAcc=[]
avgAcc2=[]
std=[]
std2=[]
stdAcc=[]
stdAcc2=[]
for i in range(epoch):
    avg.append(np.mean(np.transpose(lossList)[i]))
    avg2.append(np.mean(np.transpose(lossList2)[i]))
    std.append(np.std(np.transpose(lossList)[i]))
    std2.append(np.std(np.transpose(lossList2)[i]))
    
    avgAcc.append(np.mean(np.transpose(accList)[i]))
    avgAcc2.append(np.mean(np.transpose(accList2)[i]))
    stdAcc.append(np.std(np.transpose(accList)[i]))
    stdAcc2.append(np.std(np.transpose(accList2)[i]))
last_loss=np.transpose(lossList)[i]
last_losst=np.transpose(lossList2)[i]
last_acc=np.transpose(accList)[i]
last_acct=np.transpose(accList2)[i]
