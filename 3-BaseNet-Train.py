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

from BaseNet.BaseNet import BaseNet

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def test(denoiser, classifier, testLoader, vision_criterion=nn.MSELoss(), class_criterion=nn.CrossEntropyLoss(), noise_level=0., alpha=1.):
    total = 0.
    
    # 视觉重建数据
    vision_loss_total = 0.0
    total_mse = 0.
    total_psnr = 0.
    total_ssim = 0.
    
    # 类别重建数据
    class_loss_total = 0.0
    correct = 0.0
    
    labels_total = torch.tensor([], dtype=torch.long)
    predict_total = torch.tensor([], dtype=torch.long)

    loss_total = 0.0
    
    denoiser = denoiser.eval()
    classifier = classifier.eval()
    
    with torch.no_grad():
        for images, labels in testLoader:
            images = images.cuda()
            labels = labels.cuda()
            
            noised_images = images + torch.randn(images.size()).mul_(float(noise_level) / 255.0).cuda()
            noised_images = torch.clamp(noised_images, min=0., max=1.)
            
            denoised_images = denoiser(noised_images).clamp(min=0., max=1.)
            
            logits = classifier(denoised_images)
            predict = logits.argmax(axis=1)
            
            correct += (predict.cpu() == labels.cpu()).sum()
            labels_total = torch.cat((labels_total, labels.cpu()))
            predict_total = torch.cat((predict_total, predict.cpu()))
            
            class_loss = class_criterion(logits, labels)
            vision_loss = vision_criterion(images, denoised_images)
            loss = float(alpha) * class_loss + vision_loss
            
            class_loss_total += class_loss.item()
            vision_loss_total += vision_loss.item()
            loss_total += loss.item()
            
            mse = F.mse_loss(images, denoised_images, reduction='mean')
            total_mse += mse.item()
            total_psnr += 10 * torch.log10(1 / mse).item()
            total_ssim += ssim(images, denoised_images).item()
            total += labels.size(0)
        
        avg_mse = total_mse / len(testLoader)
        avg_psnr = total_psnr / len(testLoader)
        avg_ssim = total_ssim / len(testLoader)
        avg_class_loss = class_loss_total / len(testLoader)
        avg_vision_loss = vision_loss_total / len(testLoader)
        avg_loss = loss_total / len(testLoader)
        
        accuracy = correct.item() / total * 100.
    
    return avg_vision_loss, avg_class_loss, avg_loss, accuracy, avg_mse, avg_psnr, avg_ssim

# Save Model's Parameters
def save_model(model, path):
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


# Train Denoiser
def train(denoiser, classifier, trainLoader, testLoader, optimizer, scheduler, epochs, dataset, vision_criterion=nn.MSELoss(), 
          class_criterion=nn.CrossEntropyLoss(), noise_level=0., alpha=1., save: bool = True):
    
    Dir = r'./BaseNet/{}/'.format(dataset)
    
    if dataset not in ['mnist',  'svhn']:
        raise Exception("Invalid Dataset Name:", dataset, "DataSet supports mnist, svhn")
    
    class_best_path_gpu = Dir + dataset + r'-alpha={}-noise={}-bestclass-GPU.pth'.format(alpha, noise_level)
    vision_best_path_gpu = Dir + dataset + r'-alpha={}-noise={}-bestvision-GPU.pth'.format(alpha, noise_level)
    loss_best_path_gpu = Dir + dataset + r'-alpha={}-noise={}-bestloss-GPU.pth'.format(alpha, noise_level)
    
    class_best_path_cpu = Dir + dataset + r'-alpha={}-noise={}-bestclass-CPU.pth'.format(alpha, noise_level)
    vision_best_path_cpu = Dir + dataset + r'-alpha={}-noise={}-bestvision-CPU.pth'.format(alpha, noise_level)
    loss_best_path_cpu = Dir + dataset + r'-alpha={}-noise={}-bestloss-CPU.pth'.format(alpha, noise_level)
    
    train_log_path = Dir + r'log/trainLog-{}-alpha={}-noise={}.pt'.format(dataset, alpha, noise_level)
    test_log_path = Dir + r'log/testLog-{}-alpha={}-noise={}.pt'.format(dataset, alpha, noise_level)
    
    print("Best Loss Adv Denoiser Path-GPU : ", loss_best_path_gpu)
    print("Best Class Adv Denoiser Path-GPU : ", class_best_path_gpu)
    print("Best Vision Adv Denoiser Path-GPU : ", vision_best_path_gpu)
    
    print("Best Loss Adv Denoiser Path-CPU : ", loss_best_path_cpu)
    print("Best Class Adv Denoiser Path-CPU : ", class_best_path_cpu)
    print("Best Vision Adv Denoiser Path-CPU : ", vision_best_path_cpu)

    print("Adv Denoiser Train Log Path : ", train_log_path)
    print("Adv Denoiser Train Log Path : ", test_log_path)
    
    iteration_list = []
    
    train_vision_list = []
    train_class_list = []
    train_loss_list = []
    
    train_mse_list = []
    train_psnr_list = [] # 峰值信噪比
    train_ssim_list = [] # 结构相似度
    
    test_vision_list = []
    test_class_list = []
    test_loss_list = []
    
    test_acc_list = []
    
    test_mse_list = []
    test_psnr_list = []
    test_ssim_list = []
    
    best_class_loss = 1e10
    best_vision_loss = 1e10
    best_loss = 1e10
    
    num_iter = 0
    
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(trainLoader):
            images = images.cuda()
            labels = labels.cuda()
            
            noised_images = images + torch.randn(images.size()).mul_(float(noise_level) / 255.0).cuda()
            noised_images = torch.clamp(noised_images, min=0., max=1.)
            
            denoiser = denoiser.train()
            classifier = classifier.eval()
            optimizer.zero_grad()
            
            denoised_images = denoiser(noised_images).clamp(min=0., max=1.)
            
            logits = classifier(denoised_images)
            
            class_loss = class_criterion(logits, labels)
            vision_loss = vision_criterion(images, denoised_images)
            loss = float(alpha) * class_loss + vision_loss
            
            loss.backward()
            optimizer.step()
            
            num_iter += 1
            
            if num_iter % 50 == 0:
                iteration_list.append(num_iter)
                train_vision_list.append(vision_loss.item())
                train_class_list.append(class_loss.item())
                train_loss_list.append(loss.item())
                
                accuracy = (logits.argmax(axis=1) == labels).float().mean().item() * 100
                
                train_mse = F.mse_loss(images, denoised_images, reduction='mean')
                train_mse_list.append(train_mse.item())
                
                train_psnr = 10 * torch.log10(1. / train_mse)
                train_psnr_list.append(train_psnr.item())
                
                train_ssim = ssim(images, denoised_images)
                train_ssim_list.append(train_ssim.item())
                
                print("Train Loop:{} -- Vision:{:.10e}, Class:{:.10e}, Loss:{:.10e}, Accuracy:{}, MSE:{:.6e}, PSNR:{:.4f}, SSIM:{:.4f}".format(
                    num_iter, vision_loss.item(), class_loss.item(), loss.item(), accuracy, train_mse.item(), train_psnr.item(), train_ssim.item()))
        
        if scheduler != None:
            scheduler.step()
            

        test_vision_loss, test_class_loss, test_loss, test_acc, test_mse, test_psnr, test_ssim = test(denoiser=denoiser, 
                                                                                               classifier=classifier, 
                                                                                               testLoader=testLoader, 
                                                                                               vision_criterion=vision_criterion, 
                                                                                               class_criterion=class_criterion, 
                                                                                               noise_level=noise_level, 
                                                                                               alpha=alpha)
        
        test_vision_list.append(test_vision_loss)
        test_class_list.append(test_class_loss)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        test_mse_list.append(test_mse)
        test_psnr_list.append(test_psnr)
        test_ssim_list.append(test_ssim)
        
        print("Epoch:{} -- Test -- Vision:{:.10e}, Class:{:.10e}, Loss:{:.10e}, Accuracy:{}, MSE:{:.8e}, PSNR:{:.4f}, SSIM:{:.4f}".format(
            epoch+1, test_vision_loss, test_class_loss, test_loss, test_acc, test_mse, test_psnr, test_ssim))
        
        best_class_loss = 1e10
        best_mse_loss = 1e10
        best_loss = 1e10
        
        if test_vision_loss < best_vision_loss:
            best_vision_denoiser = denoiser
            best_vision_loss = test_vision_loss
            if save:
                save_model(model=best_vision_denoiser, path=vision_best_path_gpu)
                print("Best Vision Denoiser Updates Successfully! Test Vision Loss: {}, Accuracy: {}".format(test_vision_loss, test_acc))
        
        if test_class_loss < best_class_loss:
            best_class_denoiser = denoiser
            best_class_loss = test_class_loss
            if save:
                save_model(model=best_class_denoiser, path=class_best_path_gpu)
                print("Best Class Denoiser Updates Successfully! Test Class Loss: {}, Accuracy: {}".format(test_class_loss, test_acc))
        
        if test_loss < best_loss:
            best_loss_denoiser = denoiser
            best_loss = test_loss
            if save:
                save_model(model=best_loss_denoiser, path=loss_best_path_gpu)
                print("Best Loss Denoiser Updates Successfully! Test Loss: {}, Accuracy: {}".format(test_loss, test_acc))
        
    if save:
        save_model(model=best_class_denoiser, path=class_best_path_cpu)
        print("Best Class Denoiser Saves Successfully!")
        
        save_model(model=best_vision_denoiser, path=vision_best_path_cpu)
        print("Best Vision Denoiser Saves Successfully!")
        
        save_model(model=best_loss_denoiser, path=loss_best_path_cpu)
        print("Best Loss Denoiser Saves Successfully!")
        
        train_log = [iteration_list, train_vision_list, train_class_list, train_loss_list, train_mse_list, train_psnr_list, train_ssim_list]
        test_log = [test_vision_list, test_class_list, test_loss_list, test_acc_list, test_mse_list, test_psnr_list, test_ssim_list]
        
        torch.save(train_log, train_log_path)
        print("Train Log Saves Successfully - ", train_log_path)
        
        torch.save(test_log, test_log_path)
        print("Test Log Saves Successfully - ", test_log_path)

# Main
def main():
    # python3 BaseNet-Train.py -d mnist -a 1 -n 0 10 20 -e 10 -b 256 -g 1,2 -l -p
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, choices=['mnist', 'svhn'], nargs=1)
    parser.add_argument('-a', '--alphas', required=True, nargs='+', type=int)
    parser.add_argument('-n', '--noiseLevels', required=True, nargs='+', type=int)
    parser.add_argument('-e', '--epoch', required=True, nargs=1, type=int)
    parser.add_argument('-b', '--batchsize', required=True, nargs=1, type=int)
    parser.add_argument('-g','--gpu', required=True, nargs=1)
    parser.add_argument('-l', '--lr_scheduler', action='store_true')
    parser.add_argument('-p', '--parallel', action='store_true')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu[0]
    
    dataset = args.dataset[0]
    
    if dataset == 'mnist':
        num_channels = 1
    elif dataset == 'svhn':
        num_channels = 3
    else:
        raise Exception("Invalid dataset Name:", dataset, "DataSet supports mnist, svhn")
        
    batchsize = args.batchsize[0]
    epochs = args.epoch[0]
    
    alphas = args.alphas
    noiseLevels = args.noiseLevels
    
    sys.stdout = utils.Logger(r'./BaseNet/trainLog/trainLog-{}-{}-{}.txt'.format(dataset, alphas[0], args.gpu[0]))
    
    trainLoader = utils.getTrainDataLoader(dataset=dataset, BATCHSIZE=batchsize, shuffle=True)
    testLoader = utils.getTestDataLoader(dataset=dataset, BATCHSIZE=batchsize, shuffle=False)
    
    vision_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    for alpha in alphas:
        for noise_level in noiseLevels:
            denoiser = BaseNet(num_channels=num_channels).cuda()
            classifier = utils.getClassifier(dataset=dataset).cuda()
            
            if args.parallel == True:
                denoiser = torch.nn.DataParallel(denoiser, device_ids=list(range(torch.cuda.device_count())))
                classifier = torch.nn.DataParallel(classifier, device_ids=list(range(torch.cuda.device_count())))
            
            optimizer = torch.optim.Adam(params=denoiser.parameters(), lr=1e-3)
            
            if args.lr_scheduler == True:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200], gamma=0.1)
            elif args.lr_scheduler == False:
                scheduler = None
            
            print("Dataset :", dataset)
            print("Alpha : ", alpha)
            print("Noise Level : ", noise_level)
            print("Epoch : ", epochs)
            print("Batchsize : ", batchsize)
            print("GPU : ", args.gpu[0])
            print("Using Learning Rate Scheduler : ", args.lr_scheduler)
            print("Parallel Training : ", args.parallel)
            
            print("Testing before training...")
            test_before_train = test(denoiser=denoiser, 
                                     classifier=classifier, 
                                     testLoader=testLoader, 
                                     vision_criterion=vision_criterion, 
                                     class_criterion=class_criterion, 
                                     noise_level=noise_level, 
                                     alpha=alpha)
            
            print("Vision:{:.10e}, Class:{:.10e}, Loss:{:.10e}, Accuracy:{}, MSE:{:.8e}, PSNR:{:.4f}, SSIM:{:.4f}".format(
                test_before_train[0], test_before_train[1],test_before_train[2],test_before_train[3],test_before_train[4],test_before_train[5],test_before_train[6]))
            
            print('*****'*6 + '\t' + 'Denoiser Alpha: {} Noise Level: {} Train Begin'.format(alpha, noise_level) + '\t' + '*****'*6)
            
            train(denoiser=denoiser, 
                  classifier=classifier, 
                  trainLoader=trainLoader, 
                  testLoader=testLoader, 
                  optimizer=optimizer, 
                  scheduler=None, 
                  epochs=epochs, 
                  dataset=dataset, 
                  vision_criterion=vision_criterion, 
                  class_criterion=class_criterion, 
                  noise_level=noise_level, 
                  alpha=alpha, 
                  save=True)
            
            print('*****'*6 + '\t' + 'Denoiser Alpha: {} Noise Level: {} Train Finish'.format(alpha, noise_level) + '\t' + '*****'*6)

if __name__ == '__main__':
    main()