import os
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import utils
from model_adv import AlexNet, MoE_alexnet

def test(args):
    if args.dataset == 'cifar':
        output_classes = 10

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)

    test_loader = DataLoader(dataset['test_data'], batch_size = args.batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #model = MoE(args.num_experts, output_classes)

    model = AlexNet(output_classes)

    model = model.to(device)
    
    #print(model)

    #model = utils.cuda(model, args.gpu_ids)

    if (args.checkpoint_loc == None):
        print('Please specify a checkpoint location for the model !!!')
    
    #ckpt = utils.load_checkpoint(args.checkpoint_loc)
    #model.load_state_dict(ckpt['net'])
    #model.load_state_dict(ckpt)
    #checkpoint = torch.load(args.checkpoint_loc)
    #model.load_state_dict({k.replace('module.',''):v for k,v in checkpoint['net'].items()})
    #best_acc_nat = 

    best_acc_nat, best_acc_l2, best_acc_linf = utils.load_model(args.checkpoint_loc, model)

    model.eval()
    correct_final = 0
    for images, labels in test_loader:
        
        #images, labels = utils.cuda([images, labels], args.gpu_ids)

        images = images.to(device)
        labels = labels.to(device)

        prediction = model(images)
        pred = prediction.argmax(dim=1, keepdim=True)
        correct = pred.eq(labels.view_as(pred)).sum().item()
        correct_final = correct_final + correct
        
    
    print('original accuracy of clean images is:', best_acc_nat)
    print('original accuracy of l2 attacked images is:', best_acc_l2)
    print('original accuracy of linf attacked images is:', best_acc_linf)
    print('Final accuracy of the model is: ', correct_final / len(test_loader.dataset))
    
