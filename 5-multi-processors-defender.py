import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import pandas as pd
from math import exp
import matplotlib.pyplot as plt

import argparse
import utils
import sys
import os

import kornia
from kornia.filters import GaussianBlur2d, Laplacian, Sobel

from Defed.Defed import Defed
from utils import get_adv_data_loader, get_benign_data_loader


def getDefed(dataset, edge_method: str = 'laplacian', alpha=0., noise_level=0.):
    
    if edge_method not in ['laplacian', 'sobel']:
        raise Exception("Invalid edge_method:", edge_method, "edge_method supports laplacian, sobel")
    
    if dataset == 'mnist':
        num_channels = 1
    elif dataset == 'svhn':
        num_channels = 3
    else:
        raise Exception("Invalid dataset Name:", dataset, "DataSet supports mnist, svhn")

    denoiser = Defed(num_channels=num_channels, edge_method=edge_method)
    
    Dir = r'./Defed/{}/'.format(dataset)
    path = Dir + dataset + r'-{}-alpha={}-noise={}-bestloss-GPU.pth'.format(edge_method, alpha, noise_level)
    
    denoiser.load_state_dict(torch.load(path))
    
    return denoiser.eval()


def compute_predict_acc(defender, data_loader, adversarial=True):
    detect_result = torch.tensor([], dtype=torch.bool)
    
    correct = 0.0
    total = 0.0
    
    if adversarial:
        for benign_images, adv_images, ture_labels in data_loader:
            detect, predict = defender(adv_images)
            correct += (predict.cpu() == ture_labels.cpu()).sum()
            total += ture_labels.size(0)
            accuracy = (correct / total).item() * 100.
            
            detect_result = torch.cat((detect_result, detect))
            detect_acc = (detect_result.sum().item() / len(detect_result)) * 100.
            
    else:
        for images, labels in data_loader:
            detect, predict = defender(images)
            correct += (predict.cpu() == labels.cpu()).sum()
            total += labels.size(0)
            accuracy = (correct / total).item() * 100.
            
            detect_result = torch.cat((detect_result, detect))
            detect_acc = (1. - detect_result.sum().item() / len(detect_result)) * 100.
            
    return detect_acc, accuracy


class Defed_Processor():
    def __init__(self, denoiser):
        super(Defed_Processor, self).__init__()
        self.denoiser = denoiser.cuda().eval()
    
    def forward(self, images):
        with torch.no_grad():
            out = self.denoiser(images.cuda())
        return out.cpu()
    
    def __call__(self, x):
        return self.forward(x)


class Defender_With_Processors():
    def __init__(self, processors, classifier, alpha=1, thresholds=[None], temperature=1.):
        super(Defender_With_Processors, self).__init__()
        self.processors = processors
        self.classifier = classifier.eval().cuda()
        self.alpha = alpha
        self.thresholds = [None] * len(self.processors)
        self.temperature = temperature
    
    
    def _compute_adv_score(self, denoiser, images):
        batchsize = images.size(0)
        with torch.no_grad():
            denoised_images = denoiser(images)
            denoised_images = torch.clamp(denoised_images, min=0., max=1.).cpu()
            vision_loss = F.mse_loss(images.reshape(batchsize, -1), denoised_images.reshape(batchsize, -1), reduction='none').mean(dim=1).cpu()
            
            orign_pre = self.classifier(images.cuda()).argmax(axis=1)
            denoised_pre = self.classifier(denoised_images.cuda())
            class_loss = F.cross_entropy(denoised_pre, orign_pre, reduction='none').cpu()
            
            adv_score = self.alpha * class_loss + vision_loss
        
        return adv_score
    
    def compute_adv_score(self, images):
        adv_score_list = []
        for denoiser in self.processors:
            adv_score = self._compute_adv_score(denoiser=denoiser, images=images)
            adv_score_list.append(adv_score)
            
        return adv_score_list
    
    
    def _compute_adv_score_by_loader(self, denoiser, data_loader, adversarial=True):
        adv_score_set = torch.tensor([], dtype=torch.float32)
        
        if adversarial:
            for benign_images, adv_images, labels in data_loader:
                adv_score = self._compute_adv_score(denoiser=denoiser, images=adv_images)
                adv_score_set = torch.cat((adv_score_set, adv_score))
        else:
            for images, labels in data_loader:
                adv_score = self._compute_adv_score(denoiser=denoiser, images=images)
                adv_score_set = torch.cat((adv_score_set, adv_score))
        
        return adv_score_set
    
    
    def compute_adv_score_by_loader(self, data_loader, adversarial=True):
        adv_score_list = []
        for denoiser in self.processors:
            adv_score_set = self._compute_adv_score_by_loader(denoiser=denoiser, data_loader=data_loader, adversarial=adversarial)
            adv_score_list.append(adv_score_set)
        
        return adv_score_list
    
    
    def compute_adv_threshold(self, adv_score_list, edge=3):
        thres_list = []
        for adv_score_set in adv_score_list:
            var = adv_score_set.std()
            mean = adv_score_set.mean()
            thres = mean + edge * var
            thres_list.append(thres)

        return thres_list
    
    
    def detect(self, images):
        result = torch.tensor([], dtype=torch.bool)
        
        adv_score_list = self.compute_adv_score(images)
        del images
        
        for i in range(len(self.processors)):
            if self.thresholds[i] == None:
                raise Exception("No {}-th Threshold, Please Set Threshold First ".format(i))
            else:
                res = adv_score_list[i] > self.thresholds[i]
                result = torch.cat((result, res.reshape(1, -1)))
        
        detect_result = torch.sum(result, axis=0, dtype=torch.bool)
        
        return detect_result
    
    def classify(self, images):
        batchsize = images.size(0)
        adv_score_list = []
        denoised_logits_list = []
        denoised_softmax_list = []
        
        detect_result = torch.tensor([], dtype=torch.bool)
        
        for denoiser in self.processors:
            with torch.no_grad():
                denoised_images = denoiser(images)
                denoised_images = torch.clamp(denoised_images, min=0., max=1.).cpu()
                vision_loss = F.mse_loss(images.reshape(batchsize, -1), denoised_images.reshape(batchsize, -1), reduction='none').mean(dim=1).cpu()

                orign_pre = self.classifier(images.cuda()).argmax(axis=1)
                denoised_logits = self.classifier(denoised_images.cuda())
                class_loss = F.cross_entropy(denoised_logits, orign_pre, reduction='none').cpu()

                adv_score = self.alpha * class_loss + vision_loss
                
                adv_score_list.append(adv_score)
                denoised_logits_list.append(denoised_logits.cpu())
                
                # denoised_softmax = torch.softmax(denoised_logits / self.temperature, dim=1).cpu()
                denoised_softmax = torch.softmax(denoised_logits, dim=1).cpu()
                denoised_softmax_list.append(denoised_softmax)

        adv_softmax_result = torch.zeros(batchsize, denoised_logits.size(1), dtype=denoised_logits.dtype)
        
        
        for i in range(len(self.processors)):
            if self.thresholds[i] == None:
                raise Exception("No {}-th Threshold, Please Set Threshold First !!!".format(i))
            else:
                detect_res = adv_score_list[i] > self.thresholds[i]
                detect_result = torch.cat((detect_result, detect_res.reshape(1, -1)))
                
                adv_index = torch.where(detect_res==True)
                adv_softmax_result[adv_index] = adv_softmax_result[adv_index] + denoised_softmax_list[i][adv_index]
        
        
        detect_result = torch.sum(detect_result, axis=0, dtype=torch.bool)
        
        denoised_class = adv_softmax_result.argmax(axis=1).cpu().detach()
        origin_class = orign_pre.cpu().detach()
        
        class_result = torch.tensor([-1]*origin_class.size(0), dtype=origin_class.dtype)
        
        # 良性样本类别为原始图片的预测类别
        class_result[torch.where(detect_result==False)] = origin_class[torch.where(detect_result==False)]
        
        # 对抗样本的类别为降噪后图片的预测类别
        class_result[torch.where(detect_result==True)] = denoised_class[torch.where(detect_result==True)]

        return detect_result, class_result
    
    
    def __call__(self, x):
        return self.classify(x)

    
def main():
    
    epsilons = {
    'inf' : [8./256., 16./256., 32./256., 64./256., 80./256., 128./256.],
    'l1' : [5, 10, 15, 20, 25, 30, 40],
    'l2' : [0.125, 0.25, 0.3125, 0.5, 1, 1.5, 2],
}
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, choices=['mnist', 'svhn'], nargs=1)
    parser.add_argument('-s', '--structure', required=True, choices=['sobel', 'laplacian'], nargs=1)
    parser.add_argument('-e', '--edge', required=True, choices=[1, 2, 3], nargs=1, type=int) # 2 or 3
    parser.add_argument('-g','--gpu', required=True, nargs=1)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu[0]
    dataset = args.dataset[0]
    if dataset == 'mnist':
        noise_level = [0, 10, 20, 40, 50, 60, 80, 100, 120, 160]
    elif dataset == 'svhn':
        noise_level = [20, 40, 60, 80]          #[0,10,20,40,60,80]
    else:
        raise Exception("Invalid dataset Name:", dataset, "DataSet supports mnist, svhn")
    
    alpha = 1
    batchsize = 32
    edge = args.edge[0]
    edge_method = args.structure[0]

    classifier = utils.getClassifier(dataset=dataset).cpu().eval()
    benign_loader = get_benign_data_loader(dataset=dataset, batch_size=batchsize, shuffle=False)

    accuracy_dic = {}

    for noise_1 in noise_level:
        for noise_2 in noise_level:
            print(noise_1, noise_2)

            denoiser_1 = getDefed(dataset=dataset, edge_method=edge_method, alpha=alpha, noise_level=noise_1).eval()
            denoiser_2 = getDefed(dataset=dataset, edge_method=edge_method, alpha=alpha, noise_level=noise_2).eval()

            processor_1 = Defed_Processor(denoiser=denoiser_1)
            processor_2 = Defed_Processor(denoiser=denoiser_2)
            processors = [processor_1, processor_2]

            defender = Defender_With_Processors(processors=processors, classifier=classifier, alpha=alpha)

            benign_score_list = defender.compute_adv_score_by_loader(data_loader=benign_loader, adversarial=False)
            defender.thresholds = defender.compute_adv_threshold(adv_score_list=benign_score_list, edge=edge)

            # Benign Images
            detect_acc, acc = compute_predict_acc(defender=defender, data_loader=benign_loader, adversarial=False)

            idx = "{}-{}-{}".format(noise_1, noise_2, 'detect')
            accuracy_dic[idx] = [detect_acc]

            idx = "{}-{}-{}".format(noise_1, noise_2, 'classify')
            accuracy_dic[idx] = [acc]

            # CW-Inf
            for eps in epsilons['inf']:
                attack_method = 'CWInf-Eps={}'.format(eps)
                adv_loader = get_adv_data_loader(dataset=dataset, attack_method=attack_method, batch_size=batchsize, shuffle=False)
                detect_acc, acc = compute_predict_acc(defender=defender, data_loader=adv_loader, adversarial=True)
                idx = "{}-{}-{}".format(noise_1, noise_2, 'detect')
                accuracy_dic[idx].append(detect_acc)
                idx = "{}-{}-{}".format(noise_1, noise_2, 'classify')
                accuracy_dic[idx].append(acc)

            # JSMA
            attack_method = 'JSMA'
            adv_loader = get_adv_data_loader(dataset=dataset, attack_method=attack_method, batch_size=batchsize, shuffle=False)
            detect_acc, acc = compute_predict_acc(defender=defender, data_loader=adv_loader, adversarial=True)
            idx = "{}-{}-{}".format(noise_1, noise_2, 'detect')
            accuracy_dic[idx].append(detect_acc)
            idx = "{}-{}-{}".format(noise_1, noise_2, 'classify')
            accuracy_dic[idx].append(acc)

            # PGD-L1
            for eps in epsilons['l1']:
                attack_method = 'PGDL1-Eps={}'.format(eps)
                adv_loader = get_adv_data_loader(dataset=dataset, attack_method=attack_method, batch_size=batchsize, shuffle=False)
                detect_acc, acc = compute_predict_acc(defender=defender, data_loader=adv_loader, adversarial=True)
                idx = "{}-{}-{}".format(noise_1, noise_2, 'detect')
                accuracy_dic[idx].append(detect_acc)
                idx = "{}-{}-{}".format(noise_1, noise_2, 'classify')
                accuracy_dic[idx].append(acc)

            # PGD-L2
            for eps in epsilons['l2']:
                attack_method = 'PGDL2-Eps={}'.format(eps)
                adv_loader = get_adv_data_loader(dataset=dataset, attack_method=attack_method, batch_size=batchsize, shuffle=False)
                detect_acc, acc = compute_predict_acc(defender=defender, data_loader=adv_loader, adversarial=True)
                idx = "{}-{}-{}".format(noise_1, noise_2, 'detect')
                accuracy_dic[idx].append(detect_acc)
                idx = "{}-{}-{}".format(noise_1, noise_2, 'classify')
                accuracy_dic[idx].append(acc)

            # DeepFool
            attack_method = 'DeepFool'
            adv_loader = get_adv_data_loader(dataset=dataset, attack_method=attack_method, batch_size=batchsize, shuffle=False)
            detect_acc, acc = compute_predict_acc(defender=defender, data_loader=adv_loader, adversarial=True)
            idx = "{}-{}-{}".format(noise_1, noise_2, 'detect')
            accuracy_dic[idx].append(detect_acc)
            idx = "{}-{}-{}".format(noise_1, noise_2, 'classify')
            accuracy_dic[idx].append(acc)

            # HopSkipJump-L2
            attack_method = 'HopSkipJumpL2'
            adv_loader = get_adv_data_loader(dataset=dataset, attack_method=attack_method, batch_size=batchsize, shuffle=False)
            detect_acc, acc = compute_predict_acc(defender=defender, data_loader=adv_loader, adversarial=True)
            idx = "{}-{}-{}".format(noise_1, noise_2, 'detect')
            accuracy_dic[idx].append(detect_acc)
            idx = "{}-{}-{}".format(noise_1, noise_2, 'classify')
            accuracy_dic[idx].append(acc)

            # HopSkipJump-Inf
            attack_method = 'HopSkipJumpInf'
            adv_loader = get_adv_data_loader(dataset=dataset, attack_method=attack_method, batch_size=batchsize, shuffle=False)
            detect_acc, acc = compute_predict_acc(defender=defender, data_loader=adv_loader, adversarial=True)
            idx = "{}-{}-{}".format(noise_1, noise_2, 'detect')
            accuracy_dic[idx].append(detect_acc)
            idx = "{}-{}-{}".format(noise_1, noise_2, 'classify')
            accuracy_dic[idx].append(acc)

            # Square Attack
            for eps in epsilons['inf']:
                attack_method = 'SquareAttack-Eps={}'.format(eps)
                adv_loader = get_adv_data_loader(dataset=dataset, attack_method=attack_method, batch_size=batchsize, shuffle=False)
                detect_acc, acc = compute_predict_acc(defender=defender, data_loader=adv_loader, adversarial=True)
                idx = "{}-{}-{}".format(noise_1, noise_2, 'detect')
                accuracy_dic[idx].append(detect_acc)
                idx = "{}-{}-{}".format(noise_1, noise_2, 'classify')
                accuracy_dic[idx].append(acc)

    df = pd.DataFrame(accuracy_dic)
    path = r'./result-{}-edge={}-method={}.csv'.format(dataset, edge, edge_method)
    df.to_csv(path)
    
if __name__ == '__main__':
    main()