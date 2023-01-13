import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
import os
import argparse

from classifier.mnist.mnist import MnistClassifier
from classifier.SVHN.svhn import SVHNClassifier

import utils

def get_benign_data(classifier, data_loader, num:int):
    benign_images = torch.tensor([])
    benign_labels = torch.tensor([], dtype=torch.long)
    classifier = classifier.cuda().eval()
    
    for images, labels in data_loader:
        b, c, w, h = images.shape
        with torch.no_grad():
            images = images.cuda()
            labels = labels.cuda()
            
            output = classifier(images)
            predict = output.argmax(axis=1)
            result = (predict == labels).cpu()
            index = torch.where(result == True)
            print("Accuracy: ", (result == True).sum().item() / b * 100)
            
            benign_images = torch.cat((benign_images, images.cpu()[index].reshape(-1, c, w, h)))
            benign_labels = torch.cat((benign_labels, labels.cpu()[index]))
            
            if len(benign_labels) >= num:
                benign_images = benign_images[:num]
                benign_labels = benign_labels[:num]
                break
    
    return (benign_images, benign_labels)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, choices=['mnist', 'svhn'], nargs=1)
    parser.add_argument('-n', '--num', required=True, nargs=1, type=int)
    parser.add_argument('-g','--gpu', required=True, nargs=1)
    args = parser.parse_args()
    
    # 指定使用的GPU编号
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu[0]
    
    dataset = args.dataset[0]
    num = args.num[0]
    
    benign_data_dir = r'./adv_test_images/'
    benign_data_path = benign_data_dir + dataset + '/benign-{}.pt'.format(dataset)
    
    classifier = utils.getClassifier(dataset=dataset).cpu().eval()
    data_loader = utils.getTestDataLoader(dataset=dataset, BATCHSIZE=128, shuffle=True)

    benign_images, benign_labels = get_benign_data(classifier=classifier, data_loader=data_loader, num=num)
    torch.save([benign_images, benign_labels], benign_data_path)

if __name__ == '__main__':
    main()