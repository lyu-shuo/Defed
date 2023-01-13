import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from classifier.mnist.mnist import MnistClassifier
from classifier.SVHN.svhn import SVHNClassifier


# 获得良性样本测试集加载器
def get_benign_data_loader(dataset, batch_size=16, shuffle=False):
    data_dir = r'./adv_test_images/{}/'.format(dataset)
    data_path = data_dir + 'benign-{}.pt'.format(dataset)
    
    benign_data = torch.load(data_path)
    benign_set = torch.utils.data.TensorDataset(benign_data[0], benign_data[1])
    benign_loader = torch.utils.data.DataLoader(benign_set, batch_size=batch_size, shuffle=shuffle)
    
    return benign_loader

# 获得对抗样本测试集加载器
def get_adv_data_loader(dataset, attack_method, batch_size=16, shuffle=False):
    data_dir = r'./adv_test_images/{}/'.format(dataset)
    data_path = data_dir + '{}-{}.pt'.format(dataset, attack_method)
    
    
    adv_data = torch.load(data_path)
    adv_set = torch.utils.data.TensorDataset(adv_data[0], adv_data[1], adv_data[2])
    adv_loader = torch.utils.data.DataLoader(adv_set, batch_size=batch_size, shuffle=shuffle)
    
    return adv_loader


# 良性样本分类准确率测试工具
def test(classifier, test_loader, criterion=nn.CrossEntropyLoss()):
    classifier = classifier.cuda().eval()

    correct = 0.0
    total = 0.0
    accuracy = 0.0
    labels_total = torch.tensor([], dtype=torch.long)
    predicted_total = torch.tensor([], dtype=torch.long)
    loss_total = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda()
            
            output = classifier(images)
            predict = output.argmax(axis=1)
            total += labels.size(0)
            
            correct += (predict.cpu() == labels.cpu()).sum()
            labels_total = torch.cat((labels_total, labels.cpu()))
            predicted_total = torch.cat((predicted_total, predict.cpu()))
            
            batch_loss = criterion(output, labels)
            loss_total += batch_loss.item()
        
        accuracy = correct.item() / total * 100
        loss = loss_total / len(test_loader)
    
    return labels_total, predicted_total, accuracy, loss

# 对抗样本分类准确率测试工具
def adv_test(classifier, adv_loader, criterion=nn.CrossEntropyLoss()):
    classifier = classifier.cuda().eval()

    correct = 0.0
    total = 0.0
    accuracy = 0.0
    labels_total = torch.tensor([], dtype=torch.long)
    predicted_total = torch.tensor([], dtype=torch.long)
    loss_total = 0.0
    
    with torch.no_grad():
        for _, images, labels in adv_loader:
            images = images.cuda()
            labels = labels.cuda()
            
            output = classifier(images)
            predict = output.argmax(axis=1)
            total += labels.size(0)
            
            correct += (predict.cpu() == labels.cpu()).sum()
            labels_total = torch.cat((labels_total, labels.cpu()))
            predicted_total = torch.cat((predicted_total, predict.cpu()))
            
            batch_loss = criterion(output, labels)
            loss_total += batch_loss.item()
        
        accuracy = correct.item() / total * 100
        loss = loss_total / len(adv_loader)
    
    return labels_total, predicted_total, accuracy, loss


# 获取训练数据加载器
def getTrainDataLoader(dataset:str, BATCHSIZE=64, shuffle=True):
    dataLoader = None
    
    dataDir = {
        'Mnist' : r'./classifier/mnist/data',
        'SVHN' : r'./classifier/SVHN/data',
    }
    
    if dataset in ['mnist', 'm']:
        trainsets = datasets.MNIST(root=dataDir['Mnist'], train=True, download=False, transform=transforms.ToTensor())
        dataLoader = torch.utils.data.DataLoader(dataset=trainsets, batch_size=BATCHSIZE, shuffle=shuffle)
        print("Train DataLoader for Mnist Loads Successfully !!!")

    elif dataset in ['svhn', 's']:
        trainsets = datasets.SVHN(root=dataDir['SVHN'], split='train', download=False, transform=transforms.ToTensor())
        dataLoader = torch.utils.data.DataLoader(dataset=trainsets, batch_size=BATCHSIZE, shuffle=shuffle)
        print("Train DataLoader for SVHN Loads Successfully !!!")

    else:
        raise Exception("Invalid dataset Name:", dataset, "DataSet supports mnist, svhn")
    
    return dataLoader


# 获取测试数据加载器
def getTestDataLoader(dataset:str, BATCHSIZE=64, shuffle=False):
    dataLoader = None
    
    dataDir = {
        'Mnist' : r'./classifier/mnist/data',
        'SVHN' : r'./classifier/SVHN/data',
    }
    
    if dataset in ['mnist', 'm']:
        testsets = datasets.MNIST(root=dataDir['Mnist'], train=False, download=False, transform=transforms.ToTensor())
        dataLoader = torch.utils.data.DataLoader(dataset=testsets, batch_size=BATCHSIZE, shuffle=shuffle)
        print("Test DataLoader for Mnist Loads Successfully !!!")

    elif dataset in ['svhn', 's']:
        testsets = datasets.SVHN(root=dataDir['SVHN'], split='test', download=False, transform=transforms.ToTensor())
        dataLoader = torch.utils.data.DataLoader(dataset=testsets, batch_size=BATCHSIZE, shuffle=shuffle)
        print("Test DataLoader for SVHN Loads Successfully !!!")

    else:
        raise Exception("Invalid dataset Name:", dataset, "DataSet supports mnist, svhn")
    
    return dataLoader


# 获取分类模型
def getClassifier(dataset:str):
    classifier = None
    modelParasDir = {
        'Mnist' : r'./classifier/mnist/mnist-classifier.pth',
        'SVHN' : r'./classifier/SVHN/svhn-classifier.pth',
    }
    
    if dataset in ['mnist', 'm']:
        classifier = MnistClassifier()
        classifier.load_state_dict(torch.load(modelParasDir['Mnist']))
        print("Classifier for Mnist Loads Successfully !!!")

    elif dataset in ['svhn', 's']:
        classifier = SVHNClassifier()
        classifier.load_state_dict(torch.load(modelParasDir['SVHN']))
        print("Classifier for SVHN Loads Successfully !!!")

    else:
        raise Exception("Invalid dataset Name:", dataset, "DataSet supports mnist, svhn")
    
    return classifier