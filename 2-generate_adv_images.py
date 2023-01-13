import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils

import art
import sys
import os
import argparse

from time import time


# 对抗样本生成器
class AdvGenerator():
    def __init__(self, dataset, model):
        super(AdvGenerator, self).__init__()
        self.verbose = False
        self.dataset = dataset
        self.data_dir = r'./adv_test_images/{}/new/'.format(dataset)
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        
        input_size = {'mnist' : (1, 28, 28),  'svhn' : (3, 32, 32)}
            
        num_class = 10
        self.classifier = art.estimators.classification.PyTorchClassifier(model=model,
                                                                          clip_values=(0, 1),
                                                                          loss=self.criterion,
                                                                          input_shape=input_size[dataset],
                                                                          nb_classes=num_class)
    
    def jsma(self, benign_loader, num=100, save=True):
        attack_method = 'JSMA'
        
        benign_images = torch.tensor([], dtype=torch.float32)
        adv_images = torch.tensor([], dtype=torch.float32)
        true_labels = torch.tensor([], dtype=torch.long)
        
        attack = art.attacks.evasion.SaliencyMapMethod(classifier = self.classifier, verbose = self.verbose)
        
        num_attack = 0
        num_success = 0
        
        print("******" * 6)
        print("{} Attack Start...".format(attack_method))
        
        start = time()
        
        for images, labels in benign_loader:
            b, c, w, h = images.shape
            adv_data_np = attack.generate(x=images)
            adv_data_tensor = torch.tensor(adv_data_np)
            num_attack += len(labels)
            
            predict = self.classifier.predict(adv_data_tensor).argmax(axis=1)
            correct = (torch.tensor(predict) == labels).cpu()

            index = torch.where(correct == False)
            num_success += (correct == False).sum().item()
            
            benign_images = torch.cat((benign_images, images.cpu()[index].reshape(-1, c, w, h)))
            adv_images = torch.cat((adv_images, adv_data_tensor.cpu()[index].reshape(-1, c, w, h)))
            true_labels = torch.cat((true_labels, labels.cpu()[index]))
            
            if len(true_labels) >= num:
                benign_images = benign_images[:num]
                adv_images = adv_images[:num]
                true_labels = true_labels[:num]
                break
            
        end = time()
        
        print("Number of Total Attack : {}".format(num_attack))
        print("Number of Successful Attack : {}".format(num_success))
        print('Successful Rate of Attack : {} %'.format(num_success / num_attack * 100))
        print("Total Execution Time : {} s".format(end - start))
        print("Average Execution Time : {} s/pic".format((end - start) / num_attack))
        
        data = [benign_images, adv_images, true_labels]
        
        if save:
            self._save(attack_method=attack_method, data=data)
        else:
            return data
        
        print("{} Attack Finish ".format(attack_method))
        print("******" * 6)
    
    def pgd_l1(self, benign_loader, eps, num=100, save=True):
        attack_method = 'PGDL1-Eps={}'.format(eps)
        
        benign_images = torch.tensor([], dtype=torch.float32)
        adv_images = torch.tensor([], dtype=torch.float32)
        true_labels = torch.tensor([], dtype=torch.long)
        
        attack = art.attacks.evasion.ProjectedGradientDescent(estimator=self.classifier, norm=1, eps=eps, eps_step=4, verbose=self.verbose)
        
        num_attack = 0
        num_success = 0
        
        print("******" * 6)
        print("{} Attack Start...".format(attack_method))
        
        start = time()
        
        for images, labels in benign_loader:
            b, c, w, h = images.shape
            adv_data_np = attack.generate(x=images)
            adv_data_tensor = torch.tensor(adv_data_np)
            num_attack += len(labels)
            
            predict = self.classifier.predict(adv_data_tensor).argmax(axis=1)
            correct = (torch.tensor(predict) == labels).cpu()
            
            index = torch.where(correct == False)
            num_success += (correct == False).sum().item()
            
            benign_images = torch.cat((benign_images, images.cpu()[index].reshape(-1, c, w, h)))
            adv_images = torch.cat((adv_images, adv_data_tensor.cpu()[index].reshape(-1, c, w, h)))
            true_labels = torch.cat((true_labels, labels.cpu()[index]))
            
            if len(true_labels) >= num:
                benign_images = benign_images[:num]
                adv_images = adv_images[:num]
                true_labels = true_labels[:num]
                break
            
        end = time()
        
        print("Number of Total Attack : {}".format(num_attack))
        print("Number of Successful Attack : {}".format(num_success))
        print('Successful Rate of Attack : {} %'.format(num_success / num_attack * 100))
        print("Total Execution Time : {} s".format(end - start))
        print("Average Execution Time : {} s/pic".format((end - start) / num_attack))
        
        data = [benign_images, adv_images, true_labels]
        
        if save:
            self._save(attack_method=attack_method, data=data)
        else:
            return data
        
        print("{} Attack Finish ".format(attack_method))
        print("******" * 6)
    
    def pgd_l2(self, benign_loader, eps, num=100, save=True):
        attack_method = 'PGDL2-Eps={}'.format(eps)
        
        benign_images = torch.tensor([], dtype=torch.float32)
        adv_images = torch.tensor([], dtype=torch.float32)
        true_labels = torch.tensor([], dtype=torch.long)
        
        attack = art.attacks.evasion.ProjectedGradientDescent(estimator=self.classifier, norm=2, eps=eps, eps_step=0.1, verbose=self.verbose)
        
        num_attack = 0
        num_success = 0
        
        print("******" * 6)
        print("{} Attack Start...".format(attack_method))
        
        start = time()
        
        for images, labels in benign_loader:
            b, c, w, h = images.shape
            adv_data_np = attack.generate(x=images)
            adv_data_tensor = torch.tensor(adv_data_np)
            num_attack += len(labels)
            
            predict = self.classifier.predict(adv_data_tensor).argmax(axis=1)
            correct = (torch.tensor(predict) == labels).cpu()
            
            index = torch.where(correct == False)
            num_success += (correct == False).sum().item()
            
            benign_images = torch.cat((benign_images, images.cpu()[index].reshape(-1, c, w, h)))
            adv_images = torch.cat((adv_images, adv_data_tensor.cpu()[index].reshape(-1, c, w, h)))
            true_labels = torch.cat((true_labels, labels.cpu()[index]))
            
            if len(true_labels) >= num:
                benign_images = benign_images[:num]
                adv_images = adv_images[:num]
                true_labels = true_labels[:num]
                break
            
        end = time()
        
        print("Number of Total Attack : {}".format(num_attack))
        print("Number of Successful Attack : {}".format(num_success))
        print('Successful Rate of Attack : {} %'.format(num_success / num_attack * 100))
        print("Total Execution Time : {} s".format(end - start))
        print("Average Execution Time : {} s/pic".format((end - start) / num_attack))
        
        data = [benign_images, adv_images, true_labels]
        
        if save:
            self._save(attack_method=attack_method, data=data)
        else:
            return data
        
        print("{} Attack Finish ".format(attack_method))
        print("******" * 6)
    
    
    def deepfool(self, benign_loader, num=100, save=True):
        attack_method = 'DeepFool'
        
        benign_images = torch.tensor([], dtype=torch.float32)
        adv_images = torch.tensor([], dtype=torch.float32)
        true_labels = torch.tensor([], dtype=torch.long)
        
        attack = art.attacks.evasion.DeepFool(classifier=self.classifier, verbose=self.verbose)
        
        num_attack = 0
        num_success = 0
        
        print("******" * 6)
        print("{} Attack Start...".format(attack_method))
        
        start = time()
        
        for images, labels in benign_loader:
            b, c, w, h = images.shape
            adv_data_np = attack.generate(x=images)
            adv_data_tensor = torch.tensor(adv_data_np)
            num_attack += len(labels)
            
            predict = self.classifier.predict(adv_data_tensor).argmax(axis=1)
            correct = (torch.tensor(predict) == labels).cpu()
            
            index = torch.where(correct == False)
            num_success += (correct == False).sum().item()
            
            benign_images = torch.cat((benign_images, images.cpu()[index].reshape(-1, c, w, h)))
            adv_images = torch.cat((adv_images, adv_data_tensor.cpu()[index].reshape(-1, c, w, h)))
            true_labels = torch.cat((true_labels, labels.cpu()[index]))

            if len(true_labels) >= num:
                benign_images = benign_images[:num]
                adv_images = adv_images[:num]
                true_labels = true_labels[:num]
                break
            
        end = time()
        
        print("Number of Total Attack : {}".format(num_attack))
        print("Number of Successful Attack : {}".format(num_success))
        print('Successful Rate of Attack : {} %'.format(num_success / num_attack * 100))
        print("Total Execution Time : {} s".format(end - start))
        print("Average Execution Time : {} s/pic".format((end - start) / num_attack))
        
        data = [benign_images, adv_images, true_labels]
        
        if save:
            self._save(attack_method=attack_method, data=data)
        else:
            return data
        
        print("{} Attack Finish ".format(attack_method))
        print("******" * 6)
    
    def cwinf(self, benign_loader, eps, num=100, save=True):
        attack_method = 'CWInf-Eps={}'.format(eps)
        
        benign_images = torch.tensor([], dtype=torch.float32)
        adv_images = torch.tensor([], dtype=torch.float32)
        true_labels = torch.tensor([], dtype=torch.long)
        
        attack = art.attacks.evasion.CarliniLInfMethod(classifier=self.classifier, max_iter=200, eps=eps, verbose=self.verbose)
        
        num_attack = 0
        num_success = 0
        
        print("******" * 6)
        print("{} Attack Start...".format(attack_method))
        
        start = time()
        
        for images, labels in benign_loader:
            b, c, w, h = images.shape
            adv_data_np = attack.generate(x=images)
            adv_data_tensor = torch.tensor(adv_data_np)
            num_attack += len(labels)
            
            predict = self.classifier.predict(adv_data_tensor).argmax(axis=1)
            correct = (torch.tensor(predict) == labels).cpu()
            
            index = torch.where(correct == False)
            num_success += (correct == False).sum().item()
            
            benign_images = torch.cat((benign_images, images.cpu()[index].reshape(-1, c, w, h)))
            adv_images = torch.cat((adv_images, adv_data_tensor.cpu()[index].reshape(-1, c, w, h)))
            true_labels = torch.cat((true_labels, labels.cpu()[index]))

            if len(true_labels) >= num:
                benign_images = benign_images[:num]
                adv_images = adv_images[:num]
                true_labels = true_labels[:num]
                break
            
        end = time()
        
        print("Number of Total Attack : {}".format(num_attack))
        print("Number of Successful Attack : {}".format(num_success))
        print('Successful Rate of Attack : {} %'.format(num_success / num_attack * 100))
        print("Total Execution Time : {} s".format(end - start))
        print("Average Execution Time : {} s/pic".format((end - start) / num_attack))
        
        data = [benign_images, adv_images, true_labels]
        
        if save:
            self._save(attack_method=attack_method, data=data)
        else:
            return data
        
        print("{} Attack Finish ".format(attack_method))
        print("******" * 6)
    
    def hopskipjump_l2(self, benign_loader, num=100, save=True):
        attack_method = 'HopSkipJumpL2'
        
        benign_images = torch.tensor([], dtype=torch.float32)
        adv_images = torch.tensor([], dtype=torch.float32)
        true_labels = torch.tensor([], dtype=torch.long)
        
        attack = art.attacks.evasion.HopSkipJump(classifier = self.classifier,
                                                 targeted = False,
                                                 norm = 2,
                                                 max_iter=0,
                                                 max_eval=100,
                                                 init_eval=10,
                                                 verbose=self.verbose)
        
        num_attack = 0
        num_success = 0
        
        iter_step = 10
        
        print("******" * 6)
        print("{} Attack Start...".format(attack_method))
        
        start = time()
        
        for images, labels in benign_loader:
            b, c, w, h = images.shape
            
            adv_data_np = np.zeros(images.shape, dtype=images.numpy().dtype)
            for i in range(4):
                adv_data_np = attack.generate(x=images, x_adv_init=adv_data_np, resume=True)
                attack.max_iter = iter_step
            
            adv_data_tensor = torch.tensor(adv_data_np)
            num_attack += len(labels)
            
            predict = self.classifier.predict(adv_data_tensor).argmax(axis=1)
            correct = (torch.tensor(predict) == labels).cpu()
            
            index = torch.where(correct == False)
            num_success += (correct == False).sum().item()
            
            benign_images = torch.cat((benign_images, images.cpu()[index].reshape(-1, c, w, h)))
            adv_images = torch.cat((adv_images, adv_data_tensor.cpu()[index].reshape(-1, c, w, h)))
            true_labels = torch.cat((true_labels, labels.cpu()[index]))
            
            if len(true_labels) >= num:
                benign_images = benign_images[:num]
                adv_images = adv_images[:num]
                true_labels = true_labels[:num]
                break
            
        end = time()
        
        print("Number of Total Attack : {}".format(num_attack))
        print("Number of Successful Attack : {}".format(num_success))
        print('Successful Rate of Attack : {} %'.format(num_success / num_attack * 100))
        print("Total Execution Time : {} s".format(end - start))
        print("Average Execution Time : {} s/pic".format((end - start) / num_attack))
        
        data = [benign_images, adv_images, true_labels]
        
        if save:
            self._save(attack_method=attack_method, data=data)
        else:
            return data
        
        print("{} Attack Finish ".format(attack_method))
        print("******" * 6)

    def hopskipjump_inf(self, benign_loader, num=100, save=True):
        attack_method = 'HopSkipJumpInf'
        
        benign_images = torch.tensor([], dtype=torch.float32)
        adv_images = torch.tensor([], dtype=torch.float32)
        true_labels = torch.tensor([], dtype=torch.long)
        
        attack = art.attacks.evasion.HopSkipJump(classifier = self.classifier,
                                                 targeted = False,
                                                 norm = 'inf',
                                                 max_iter=0,
                                                 max_eval=100,
                                                 init_eval=10,
                                                 verbose = self.verbose)
        
        num_attack = 0
        num_success = 0
        
        iter_step = 10
        
        print("******" * 6)
        print("{} Attack Start...".format(attack_method))
        
        start = time()
        
        for images, labels in benign_loader:
            b, c, w, h = images.shape
            
            adv_data_np = np.zeros(images.shape, dtype=images.numpy().dtype)
            for i in range(4):
                adv_data_np = attack.generate(x=images, x_adv_init=adv_data_np, resume=True)
                attack.max_iter = iter_step
            
            adv_data_tensor = torch.tensor(adv_data_np)
            num_attack += len(labels)
            
            predict = self.classifier.predict(adv_data_tensor).argmax(axis=1)
            correct = (torch.tensor(predict) == labels).cpu()
            
            index = torch.where(correct == False)
            num_success += (correct == False).sum().item()
            
            benign_images = torch.cat((benign_images, images.cpu()[index].reshape(-1, c, w, h)))
            adv_images = torch.cat((adv_images, adv_data_tensor.cpu()[index].reshape(-1, c, w, h)))
            true_labels = torch.cat((true_labels, labels.cpu()[index]))
            
            if len(true_labels) >= num:
                benign_images = benign_images[:num]
                adv_images = adv_images[:num]
                true_labels = true_labels[:num]
                break
            
        end = time()
        
        print("Number of Total Attack : {}".format(num_attack))
        print("Number of Successful Attack : {}".format(num_success))
        print('Successful Rate of Attack : {} %'.format(num_success / num_attack * 100))
        print("Total Execution Time : {} s".format(end - start))
        print("Average Execution Time : {} s/pic".format((end - start) / num_attack))
        
        data = [benign_images, adv_images, true_labels]
        
        if save:
            self._save(attack_method=attack_method, data=data)
        else:
            return data
        
        print("{} Attack Finish ".format(attack_method))
        print("******" * 6)
    
    def squareattack(self, benign_loader, eps, num=100, save=True):
        attack_method = 'SquareAttack-Eps={}'.format(eps)
        
        benign_images = torch.tensor([], dtype=torch.float32)
        adv_images = torch.tensor([], dtype=torch.float32)
        true_labels = torch.tensor([], dtype=torch.long)
        
        attack = art.attacks.evasion.SquareAttack(estimator=self.classifier, 
                                                  norm = 'inf',
                                                  max_iter=100,
                                                  eps=eps,
                                                  verbose=self.verbose)
        
        num_attack = 0
        num_success = 0
        
        print("******" * 6)
        print("{} Attack Start...".format(attack_method))
        
        start = time()
        
        for images, labels in benign_loader:
            b, c, w, h = images.shape
            adv_data_np = attack.generate(x=images)
            adv_data_tensor = torch.tensor(adv_data_np)
            num_attack += len(labels)
            
            predict = self.classifier.predict(adv_data_tensor).argmax(axis=1)
            correct = (torch.tensor(predict) == labels).cpu()
            
            index = torch.where(correct == False)
            num_success += (correct == False).sum().item()
            
            benign_images = torch.cat((benign_images, images.cpu()[index].reshape(-1, c, w, h)))
            adv_images = torch.cat((adv_images, adv_data_tensor.cpu()[index].reshape(-1, c, w, h)))
            true_labels = torch.cat((true_labels, labels.cpu()[index]))
            
            if len(true_labels) >= num:
                benign_images = benign_images[:num]
                adv_images = adv_images[:num]
                true_labels = true_labels[:num]
                break
            
        end = time()
        
        print("Number of Total Attack : {}".format(num_attack))
        print("Number of Successful Attack : {}".format(num_success))
        print('Successful Rate of Attack : {} %'.format(num_success / num_attack * 100))
        print("Total Execution Time : {} s".format(end - start))
        print("Average Execution Time : {} s/pic".format((end - start) / num_attack))
        
        data = [benign_images, adv_images, true_labels]
        
        if save:
            self._save(attack_method=attack_method, data=data)
        else:
            return data
        
        print("{} Attack Finish ".format(attack_method))
        print("******" * 6)
    
    def _save(self, attack_method, data):
        adv_data_path = self.data_dir + '{}-{}.pt'.format(self.dataset, attack_method)
        torch.save(data, adv_data_path)
        print("{} Saves Successfully !!!".format(adv_data_path))


# 生成对抗样本
def generate(dataset, batch_size=20, num=100):
    epsilons = {
        'inf' : [8./256., 16./256., 32./256., 64./256., 80./256., 128./256.],
        'l1' : [5, 10, 15, 20, 25, 30, 40],
        'l2' : [0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2],
    }
    
    print("Attack on {} Start".format(dataset))
    classifier = utils.getClassifier(dataset=dataset).cpu().eval()
    
    data_dir = r'./adv_test_images/'
    benign_data_path = data_dir + dataset + '/benign-{}.pt'.format(dataset)
    
    benign_data = torch.load(benign_data_path)
    benign_set = torch.utils.data.TensorDataset(benign_data[0], benign_data[1])
    benign_loader = torch.utils.data.DataLoader(benign_set, batch_size=batch_size, shuffle=False)
    
    adv_generator = AdvGenerator(dataset=dataset, model=classifier)
    
    # JSMA
    adv_generator.jsma(benign_loader=benign_loader, num=num, save=True)

    # PGD-L1
    for eps in epsilons['l1']:
        adv_generator.pgd_l1(benign_loader=benign_loader, eps=eps, num=num, save=True)

    # PGD-L2
    for eps in epsilons['l2']:
        adv_generator.pgd_l2(benign_loader=benign_loader, eps=eps, num=num, save=True)

    # DeepFool
    adv_generator.deepfool(benign_loader=benign_loader, num=num, save=True)

    # CW-Inf
    for eps in epsilons['inf']:
        adv_generator.cwinf(benign_loader=benign_loader, eps=eps, num=num, save=True)
    
    # HopSkipJump-L2
    adv_generator.hopskipjump_l2(benign_loader=benign_loader, num=num, save=True)

    # HopSkipJump-Inf
    adv_generator.hopskipjump_inf(benign_loader=benign_loader, num=num, save=True)

    # Square Attack
    for eps in epsilons['inf']:
        adv_generator.squareattack(benign_loader=benign_loader, eps=eps, num=num, save=True)

def main():
    # python generateAdvImages.py -d mnist -b 32 -n 1000 -g 1
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, choices=['mnist',  'svhn'], nargs=1)
    parser.add_argument('-b', '--batchsize', required=True, nargs=1, type=int)
    parser.add_argument('-n', '--num', required=True, nargs=1, type=int)
    parser.add_argument('-g','--gpu', required=True, nargs=1)
    args = parser.parse_args()
    
    # 指定GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu[0]
    
    dataset = args.dataset[0]
    batchsize = args.batchsize[0]
    num = args.num[0]
    
    generate(dataset=dataset, batch_size=batchsize, num=num)

if __name__ == '__main__':
    main()