#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 22:49:39 2020

@author: chad
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
  transforms.ToTensor()]) 
 
unloader = transforms.ToPILImage()


## PIL to tensor

# 输入PIL格式图片
# 返回tensor变量
def PIL_to_tensor(image):
 # image = loader(image).unsqueeze(0)
  image = loader(image)
  return image.to(device, torch.float)

# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
  image = tensor.cpu().clone()
  #image = image.squeeze(0)
  image = unloader(image)
  return image