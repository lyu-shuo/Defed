import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入模型
from svhn import SVHNClassifier

# 训练超参数
BATCH_SIZE = 256
EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入数据并检查
trainsets = datasets.SVHN(root='./data', split='train', download=False, transform=transforms.ToTensor())
testsets = datasets.SVHN(root='./data', split='test', download=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=trainsets, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testsets, batch_size=BATCH_SIZE, shuffle=False)

sample_image, sample_label = next(iter(train_loader))
print(sample_image.data.shape, sample_image.data.min(), sample_image.max())
print(sample_image.data.mean(), sample_image.data.std())


# 构建分类模型
model = SVHNClassifier()
model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
print(model)

# 训练过程设置
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

def test(model, test_loader, criterion):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE).eval()
    correct = 0.0
    total = 0.0
    accuracy = 0.0
    labels_total = torch.tensor([], dtype=torch.long)
    predicted_total = torch.tensor([], dtype=torch.long)
    loss_total = 0.0
    
    with torch.no_grad():
    
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            output = model(images)
            predict = torch.max(output.data, axis=1)[1]
            total += labels.size(0)
            correct += (predict.cpu() == labels.cpu()).sum()
            
            labels_total = torch.cat((labels_total, labels.cpu()))
            predicted_total = torch.cat((predicted_total, predict.cpu()))
            
            #Loss
            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
            loss_total += batch_loss.item()
            
        accuracy = correct.item() / total * 100
        loss = loss_total / len(test_loader)
    
    return labels_total, predicted_total, accuracy, loss

# 测试初始模型
print('Test origin model loading...\n', test(model, test_loader, criterion))

def train(model, train_loader, test_loader, criterion, optimizer, EPOCHS):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_list = []
    accuracy_list = []
    iteration_list = []
    best_accuracy = 0.0
    
    test_acc_list = []
    test_loss_list = []    
    
    iter = 0
    for epoch in range(EPOCHS):
        print("--------------- Train Epoch : {} ---------------".format(epoch+1))
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            model = model.to(DEVICE).train()
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            iter += 1
            
            if iter % 200 == 0:
                accuracy = (torch.argmax(output.data, axis=1) == labels).sum() * 100. / labels.size(0)
                loss_list.append(loss.item())
                accuracy_list.append(accuracy.item())
                iteration_list.append(iter)
                
                print("Train Loop : {}, Train Loss : {}, Train Accuracy: {}".format(iter, loss.item(), accuracy))
        
        _, _, current_accuracy, current_loss = test(model, test_loader, criterion)
        test_acc_list.append(current_accuracy)
        test_loss_list.append(current_loss)
        
        print("Epoch : ", epoch+1, "Test Loss : ", current_loss, "Test Accuracy : ", current_accuracy)
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_model = model.cpu()
            print("Best model updates successfully!")
    
    model_dir = r'./svhn-classifier.pth'
    trainLog_dir = r'./log/TrainLog-svhn.pt'
    testLog_dir = r'./log/TestLog-svhn.pt'
    
    torch.save(best_model.state_dict(), model_dir)
    torch.save([iteration_list, accuracy_list, loss_list], trainLog_dir)
    torch.save([test_acc_list, test_loss_list], testLog_dir)
    print("Save Successful!")

train(model=model, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, EPOCHS=EPOCHS)