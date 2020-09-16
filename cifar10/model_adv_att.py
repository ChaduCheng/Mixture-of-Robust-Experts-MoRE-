#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:23:17 2020

@author: chad
"""

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import utils

num_classes = 10




class AlexNet(nn.Module):
    def __init__(self, output_dim):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x




def project(x, original_x, epsilon, _type='linf'):

    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)

    elif _type == 'l2':
        dist = (x - original_x)

        dist = dist.view(x.shape[0], -1)

        dist_norm = torch.norm(dist, dim=1, keepdim=True)

        mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)

        # dist = F.normalize(dist, p=2, dim=1)

        dist = dist / dist_norm

        dist *= epsilon

        dist = dist.view(x.shape)

        x = (original_x + dist) * mask.float() + x * (1 - mask.float())


    return x

# Model
class AttackPGD(nn.Module):  # change here to build l_2 and l_inf
    def __init__(self, basic_net, config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self._type = config['_type']
        assert config['loss_func'] == 'xent', 'Only xent supported for now.'

    def forward(self, inputs, targets, attack):
        if not attack:
            return self.basic_net(inputs), inputs

        x = inputs.detach()
        if self.rand:     # attack here l_inf attack
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.basic_net(x)
                #loss = F.cross_entropy(logits, targets, size_average=False)
                loss = F.cross_entropy(logits, targets, reduction='sum')
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size*torch.sign(grad.detach())
           
           # normal l_infite
           # x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            
            x = project(x, inputs, self.epsilon, self._type)
            x = torch.clamp(x, 0, 1)

        #return self.basic_net(x), x
        return x

class AlexNet_expert(nn.Module):
    def __init__(self, output_dim):
        super(AlexNet_expert, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        for p in self.parameters():    # fc  could train
            p.requires_grad=False
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, output_dim),
        )
        


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x




    
    
class MoE_alexnet(nn.Module):
    def __init__(self, num_experts, output_dim):
        super(MoE_alexnet, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.softmax = nn.Softmax()

        self.experts = nn.ModuleList([AlexNet_expert(output_dim) for i in range(num_experts)])

        self.gating = AlexNet(num_experts)

    def forward(self, x):
        
        out_final = []
        weights = self.softmax(self.gating(x))    ### Outputs a tensor of [batch_size, num_experts]
        # print(x)
        # print(len(x))
        # print(x.shape)
        # print(weights)
        # print(len(weights))
        # print(weights.shape)
        out = torch.zeros([weights.shape[0], self.output_dim])
        for i in range(self.num_experts):
            #out += weights[:, i].unsqueeze(1) * self.experts[i](x)    ### To get the output of experts weighted by the appropriate gating weight
            out = weights[:, i].unsqueeze(1) * self.experts[i](x)
            
            # print('out is :', out)
            
            out_final.append(out)
            
            # print('all out are:' , out_final)
            
            # print('size of out:', len(out_final))
            

        return sum(out_final)    ### out will have the shape [batch_size, output_dim]
