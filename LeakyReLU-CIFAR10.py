# -*- coding: utf-8 -*-
#Import needed packages
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import torch.nn.functional as F
from frac_init import yuanhan_normal_, randomwalk_normal_

# To use different initialization method, one can decomment the code

class simpleNet(nn.Module):
    
    def __init__(self, n_hidden_2, out_dim):
        super(simpleNet, self).__init__()
        
        # 1 CNN + 20 layer-Fully connected network with the new initialization method
        # We choose the s = 1 for Leaky ReLU activation
        
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = nn.Linear(16 * 5 * 5, 64)
        yuanhan_normal_(self.layer1.weight,nonlinearity='leaky_relu', s=1)
        self.layer2 = nn.Linear(64, n_hidden_2)
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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = nn.Linear(16 * 5 * 5, 64)
        nn.init.kaiming_normal_(self.layer1.weight,nonlinearity='leaky_relu')
        self.layer2 = nn.Linear(64, n_hidden_2)
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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = nn.Linear(16 * 5 * 5, 64)
        nn.init.xavier_normal_(self.layer1.weight)
        self.layer2 = nn.Linear(64, n_hidden_2)
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
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
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

#Define transformations for the training set, flip the images randomly, crop out and apply mean and std normalization
train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 512

#Load the training set
train_set = CIFAR10(root="./data",train=True,transform=train_transformations,download=True)

#Create a loder for the training set
train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=4)


#Define transformations for the test set
test_transformations = transforms.Compose([
   transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))

])

#Load the test set, note that train is set to False
test_set = CIFAR10(root="./data",train=False,transform=test_transformations,download=True)

#Create a loder for the test set, note that both shuffle is set to false for the test loader
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False,num_workers=4)


#Create a learning rate adjustment function that divides the learning rate by 10 every 30 epochs
def adjust_learning_rate(epoch):

    lr = 0.01

    if epoch > 180:
        lr = lr / 1000000
    elif epoch > 150:
        lr = lr / 100000
    elif epoch > 120:
        lr = lr / 10000
    elif epoch > 90:
        lr = lr / 1000
    elif epoch > 60:
        lr = lr / 100
    elif epoch > 30:
        lr = lr / 10

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr




def save_models(epoch):
    torch.save(model.state_dict(), "cifar10model_{}.model".format(epoch))
    print("Checkpoint saved")

def test():
    model.eval()
    test_acc = 0.0
    test_loss=0.0
    for i, (images, labels) in enumerate(test_loader):
      
        if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

        #Predict classes using images from the test set
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        #Backpropagate the loss
        loss.backward()

        test_loss+=loss.cpu().item() * images.size(0)
        _,pred = torch.max(outputs.data, 1)
        test_acc += pred.eq(labels.view_as(pred)).sum().item()
        


    #Compute the average acc and loss over all 10000 test images
    test_acc = test_acc / 10000
    test_loss=test_loss / 10000

    return test_acc,test_loss

def train(num_epochs):
    eachTrainLoss=[]
    eachTrainAcc=[]
    test_accList=[]
    test_lossList=[]
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            #Move images and labels to gpu if available
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            #Clear all accumulated gradients
            optimizer.zero_grad()
            #Predict classes using images from the test set
            outputs = model(images)
            #Compute the loss based on the predictions and actual labels
            loss = loss_fn(outputs,labels)
            #Backpropagate the loss
            loss.backward()

            #Adjust parameters according to the computed gradients
            optimizer.step()

            train_loss += loss.cpu().item() * images.size(0)
            _, pred = torch.max(outputs.data, 1)
            train_acc += pred.eq(labels.view_as(pred)).sum().item()
        #Call the learning rate adjustment function
        adjust_learning_rate(epoch)

        #Compute the average acc and loss over all 50000 training images
        train_acc = train_acc / 50000
        train_loss = train_loss / 50000
        eachTrainLoss.append(train_loss)
        eachTrainAcc.append(train_acc)
        #Evaluate on the test set
        test_acc, test_loss = test()
        test_accList.append(test_acc)
        test_lossList.append(test_loss)

        # Save the model if the test acc is greater than our current best
        # if test_acc > best_acc:
        #     save_models(epoch)
        #     best_acc = test_acc
        # Print the metrics
        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(epoch, train_acc, train_loss,test_acc))
    return eachTrainAcc,eachTrainLoss,test_accList,test_lossList
if __name__ == "__main__":
    batch_size = 512
    cuda_avail = torch.cuda.is_available()
    lossList=[]
    testList=[]
    trainacc=[]
    testacc=[]
    for i in range(10):
        batch_size = 512
        model = simpleNet(64,10)

        if cuda_avail:
            model.cuda()

        optimizer = SGD(model.parameters(), lr=0.01,weight_decay=0.0001)
        loss_fn = nn.CrossEntropyLoss()
        a,l,ta,tl=train(50)
        lossList.append(l)
        testList.append(tl)
        trainacc.append(a)
        testacc.append(ta)
    print(l)
    avg=[]
    avg2=[]
    std=[]
    std2=[]
    avgAcc=[]
    avgAcc2=[]
    stdAcc=[]
    stdAcc2=[]
    for i in range(50):
        avg.append(np.mean(np.transpose(lossList)[i]))
        avg2.append(np.mean(np.transpose(testList)[i]))
        std.append(np.std(np.transpose(lossList)[i]))
        std2.append(np.std(np.transpose(testList)[i]))
    
        avgAcc.append(np.mean(np.transpose(trainacc)[i]))
        avgAcc2.append(np.mean(np.transpose(testacc)[i]))
        stdAcc.append(np.std(np.transpose(trainacc)[i]))
        stdAcc2.append(np.std(np.transpose(testacc)[i]))
