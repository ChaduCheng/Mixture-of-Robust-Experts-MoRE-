import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models

num_classes = 10

class LeNet(nn.Module):

    # network structure
    def __init__(self, output_dim):
        super(LeNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, output_dim)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
       # x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = x.view(-1, self.num_flat_features(x))
        x = x.view(x.size()[0], -1)
        #print(x)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)


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


class MoE_lenet(nn.Module):
    def __init__(self, num_experts, output_dim):
        super(MoE_lenet, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.softmax = nn.Softmax()

        self.experts = nn.ModuleList([LeNet(output_dim) for i in range(num_experts)])

        self.gating = LeNet(num_experts)

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
    
    
class MoE_alexnet(nn.Module):
    def __init__(self, num_experts, output_dim):
        super(MoE_alexnet, self).__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim
        self.softmax = nn.Softmax()

        self.experts = nn.ModuleList([AlexNet(output_dim) for i in range(num_experts)])

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