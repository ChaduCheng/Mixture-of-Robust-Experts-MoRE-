#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 15:54:37 2020

@author: chad
"""

import os
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import optim
import torch.backends.cudnn as cudnn

import utils
from model_adv import AlexNet, MoE_alexnet
from model_adv_att import AttackPGD
from model_resnet import *
import string


def train_clean (train_loader, device,optimizer,model,CE_loss, lr_schedule, epoch_i):
    a = 0
    j = 0
    for j in range(1,3):    
        print('Doing clean images training No. ' + str(j))
        for images, labels in tqdm(train_loader):
        #for images, labels in train_loader:
    
                images = images.to(device)
                labels = labels.to(device)
    
                # optimizer.zero_grad()
                prediction, weights = model(images)
                
                #print('prediction value is :', prediction )
    
                loss = CE_loss(prediction, labels)
                # print('loss value is :', loss )
                
                lr = lr_schedule(epoch_i + (a+1)/len(train_loader))
                optimizer.param_groups[0].update(lr=lr)
    
                optimizer.zero_grad()
                
                loss.backward()
    
                optimizer.step()
                
                a = a+1

    return weights
def train_adv(train_loader, device,optimizer, basic_model, model, AttackPGD ,CE_loss,config,attack, lr_schedule, epoch_i):
    b = 0
    j = 0
    for i in config:
        j = j+1
        print('Adv Training ' + str(config[i]['_type']) + '  epsilon:' + str(j))
        
        for images_adv, labels in tqdm(train_loader):
        #for images_adv, labels in train_loader:
    
                
                net_attack = AttackPGD(basic_model,config[i])
                
                #print(net_attack)
                
                net_attack = net_attack.to(device)
                images_adv = images_adv.to(device)
                labels = labels.to(device)
                #print(images.device)
                # images.cuda(args.gpu_ids[0])
                # labels.cuda(args.gpu_ids[0])
                #print(images)
                #print(images.shape)
                #print(type(images))
                
                images_att = net_attack(images_adv,labels, attack)
                
    
    
                #optimizer.zero_grad()
                prediction, weights = model(images_att)
                
                #print('prediction value is :', prediction )
    
                loss = CE_loss(prediction, labels)
    
                lr = lr_schedule(epoch_i + (b+1)/len(train_loader))
                optimizer.param_groups[0].update(lr=lr)
    
                optimizer.zero_grad()
                # print('loss value is :', loss )
                loss.backward()
    
                optimizer.step()
                
                b = b+1

    return weights
def val(val_loader, device, model,  basic_model, AttackPGD, config_l2, config_linf, attack,\
        correct_final_nat, best_acc_nat, correct_final_l2, best_acc_l2, correct_final_linf,\
            best_acc_linf, best_acc_aver, checkpoint_loc):
    
    acc_linf = []
    acc_l2 = []
    
    print('Valuation clean images')
    
    for images_1, labels in tqdm(val_loader):
            
 
            images_1 = images_1.to(device)
            labels = labels.to(device)
            
            #images_att = net_attack(images,labels, attack)
    
            prediction, weights_nat = model(images_1)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_1 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final_nat = correct_final_nat + correct_1
            
    acc_nat = correct_final_nat / len(val_loader.dataset)
    
    j=0
    
    for i in config_l2:
        j = j + 1
        #print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
        print('Valuation l_2  epsilon: 50+ ' + str(j) + '*10 / 255')
        
        correct_2 = 0
        correct_final_l2 = 0
        
        for images_2, labels in tqdm(val_loader):
                
                #a = a+1
                net_attack = AttackPGD(basic_model,config_l2[i])
                
                #print(net_attack)
                
                net_attack = net_attack.to(device)
                #print('processing testing image:', a)
                #images, labels = utils.cuda([images, labels], args.gpu_ids)
                images_2 = images_2.to(device)
                labels = labels.to(device)
                
                images_att = net_attack(images_2,labels, attack)
    
                prediction,weights_l2 = model(images_att)
                pred = prediction.argmax(dim=1, keepdim=True)
                correct_2 = pred.eq(labels.view_as(pred)).sum().item()
                #correct_final.append(correct)
                correct_final_l2 = correct_final_l2 + correct_2
            
        acc_l2.append(correct_final_l2 / len(val_loader.dataset))
    
    j = 0
    for i in config_linf:
        
        j = j+1
    
        #print('Valuation' + str(config_linf[i]['_type']) + '  epsilon:' + str(config_linf[i]['epsilon']))
        print('Valuation l_inf  epsilon: 5+ ' + str(j) + '*10 / 255')
        correct_2 = 0
        correct_final_linf = 0
        for images_3, labels in tqdm(val_loader):
                
                #a = a+1
                net_attack = AttackPGD(basic_model,config_linf[i])
                
                #print(net_attack)
                
                net_attack = net_attack.to(device)
                #print('processing testing image:', a)
                #images, labels = utils.cuda([images, labels], args.gpu_ids)
                images_3 = images_3.to(device)
                labels = labels.to(device)
                
                images_att = net_attack(images_3,labels, attack)
    
                prediction, weights_linf = model(images_att)
                pred = prediction.argmax(dim=1, keepdim=True)
                correct_3 = pred.eq(labels.view_as(pred)).sum().item()
                #correct_final.append(correct)
                correct_final_linf = correct_final_linf + correct_3
            
        acc_linf.append(correct_final_linf / len(val_loader.dataset))    
    
    
    
    
    if (acc_nat*2 + sum(acc_l2) + sum(acc_linf))/6 > best_acc_aver:
            print('saving..')
            
            state = {
            'net': model.state_dict(),
            'acc_clean': acc_nat,
            'acc_l2': acc_l2,
            'acc_linf': acc_linf,
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc_nat = acc_nat
            best_acc_l2 = acc_l2
            best_acc_linf = acc_linf
            best_acc_aver = (best_acc_nat*2 + sum(best_acc_l2) + sum(best_acc_linf))/6
    
    return acc_nat, best_acc_nat, acc_l2, best_acc_l2, acc_linf, best_acc_linf, best_acc_aver


def train(args):
    
    config_linf_5 = {
    'epsilon': 5.0 / 255,
    #'epsilon': 0.314,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'linf'
     }
    config_linf_6 = {
    'epsilon': 6.0 / 255,
    #'epsilon': 0.314,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'linf'
     }
    config_linf_7 = {
    'epsilon': 7.0 / 255,
    #'epsilon': 0.314,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'linf'
     }
    config_linf_8 = {
    'epsilon': 8.0 / 255,
    #'epsilon': 0.314,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'linf'
     }

    config_linf = dict(config_linf_6 = config_linf_6, config_linf_8 = config_linf_8)

    config_l2_50 = {
    #'epsilon': 8.0 / 255,
    'epsilon': 50 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'l2'
     }
    
    config_l2_60 = {
    #'epsilon': 8.0 / 255,
    'epsilon': 60 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'l2'
     }
    
    config_l2_70 = {
    #'epsilon': 8.0 / 255,
    'epsilon': 70 / 255,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'l2'
     }
    


    
    config_l2_80 = {
    #'epsilon': 8.0 / 255,
    'epsilon': 0.314,
    'num_steps': 10,
    'step_size': 2.0 / 255,
    'random_start': True,
    'loss_func': 'xent',
    '_type': 'l2'
     }

    
    
    config_l2 = dict(config_l2_60 = config_l2_60, config_l2_80 = config_l2_80)
    attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10
        
    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat = 0
    best_acc_l2 = []
    best_acc_linf = []
    best_acc_aver = 0
    

    transform = utils.get_transformation(args.dataset)
    dataset = utils.get_dataset(args.dataset, transform, args.train_split)
    
    # operate this train_loader to generate new loader
    
    train_loader = DataLoader(dataset['train_data'], batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset['val_data'], batch_size = args.batch_size, shuffle=True)
    
    # l2 attack and linf attack to generate new data
    
    # for images, labels in train_loader:
    #     print(images)
    #     print(images.type)
    #     print(images.shape)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    CE_loss = nn.CrossEntropyLoss()
    # jiang LeNet dan du chan fen le chu lai
    #model =  LeNet(output_classes)
    basic_model =  ResNet18(output_classes)
    basic_model =  basic_model.to(device)
    #print(output_classes.device)
    #model = MoE(basic_model, output_classes, output_classes)
    model = MoE_ResNet18(args.num_experts, output_classes)
    
    model = model.to(device)
    
    ### load pretrained model
    
    model_loc0 = './trained_model/ckptnat_resnet18_cifar.pth'
    model_loc1 = './trained_model/ckptl2_resnet18_cifar_60_255.pth'
    model_loc2 = './trained_model/ckptl2_resnet18_cifar_80_255.pth'
    model_loc3 = './trained_model/ckptlinf_resnet18_cifar_6_255.pth'
    model_loc4 = './trained_model/ckptlinf_resnet18_cifar_8_255.pth'
    # model_loc5 = './trained_model/ckptl2_alexnet_cifar_90_255.pth'
    # model_loc6 = './trained_model/ckptl2_alexnet_cifar_100_255.pth'
    # model_loc7 = './trained_model/ckptl2_alexnet_cifar_110_255.pth'
    # model_loc8 = './trained_model/ckptlinf_alexnet_cifar_5_255.pth'
    # model_loc9 = './trained_model/ckptlinf_alexnet_cifar_6_255.pth'
    # model_loc10 = './trained_model/ckptlinf_alexnet_cifar_7_255.pth'
    # model_loc11 = './trained_model/ckptlinf_alexnet_cifar_8_255.pth'
    # model_loc12 = './trained_model/ckptlinf_alexnet_cifar_9_255.pth'
    # model_loc13 = './trained_model/ckptlinf_alexnet_cifar_10_255.pth'
    # model_loc14 = './trained_model/ckptlinf_alexnet_cifar_11_255.pth'
   #  model_loc3 = './trained_model/ckptl2_alex_cifar_50.pth'
    
    
   # #  print(model)
   # # # print(model.state_dict())
   # #  print(model.gating)
   # #  print(model.experts[0])
   # #  print(model.experts[1])
   # #  print(model.experts[2])
   # #  print(model.state_dict().keys())
    
    utils.load_model(model_loc0, model.experts[0])
    #utils.load_model(model_loc0, model.experts[1])
    utils.load_model(model_loc1, model.experts[1])
    utils.load_model(model_loc2, model.experts[2])
    utils.load_model(model_loc3, model.experts[3])
    utils.load_model(model_loc4, model.experts[4])
    # utils.load_model(model_loc5, model.experts[5])
    # utils.load_model(model_loc6, model.experts[6])
    # utils.load_model(model_loc7, model.experts[7])
    # utils.load_model(model_loc8, model.experts[5])
    # utils.load_model(model_loc9, model.experts[6])
    # utils.load_model(model_loc10, model.experts[7])
    # utils.load_model(model_loc11, model.experts[8])
    # utils.load_model(model_loc12, model.experts[12])
    # utils.load_model(model_loc13, model.experts[13])
    # utils.load_model(model_loc14, model.experts[14])
    
    
    #utils.load_model(model_loc1, model)
    # key_word = 'experts'
    
    # for name,value in model.named_parameters():
    #     if key_word in name:
    #         value.requires_grad = False
    
    #print(model.device)
    
    #print(basic_model.device)
    
    #model = MoE(args.num_experts, output_classes)
    
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    #model = utils.cuda(model, args.gpu_ids)
    #model.cuda(args.gpu_ids[0])
    #model = model.to(device)
    #model = torch.nn.DataParallel(model, device_ids = [0]).cuda()
    #print(model.device)
    #model = MoE(basic_model, args.num_experts, output_classes)
    #print(model)
    #print(type(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_schedule = lambda t: np.interp([t], [0, args.epochs*2//5, args.epochs*4//5, args.epochs], [0, 0.1, 0.005, 0])[0]


    for i in range(args.epochs):
        #model.train()
        #j = 0
        print('The epoch number is: ' + str(i))
        
        
        # train_clean (train_loader, device,optimizer,model,CE_loss)
        
        # # #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        # # #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        # train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_l2, attack)
                
        # train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_linf, attack)

        weights_nat = train_clean (train_loader, device,optimizer,model,CE_loss, lr_schedule, i)
        
        print('nat training weights', weights_nat)


        
        #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        weights_l2 = train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_l2, attack, lr_schedule, i)
        
        print('l2 training weights', weights_l2)     


        weights_linf = train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_linf, attack, lr_schedule, i)
        
        print('after linf training weights(final):', weights_linf)         
       

        # for images, labels in train_loader:
        #     #j = j + 1
            
        #     #print('processing image No.:', j)
        #     #images, labels = utils.cuda([images, labels], args.gpu_ids)
        #     #images, labels = images.to(device), labels.to(device)
            
        #     #print(images)
            
        #     net_attack = AttackPGD(basic_model,config_linf)
            
        #     #print(net_attack)
            
        #     net_attack = net_attack.to(device)
        #     images = images.to(device)
        #     labels = labels.to(device)
        #     #print(images.device)
        #     # images.cuda(args.gpu_ids[0])
        #     # labels.cuda(args.gpu_ids[0])
        #     #print(images)
        #     #print(images.shape)
        #     #print(type(images))
            
        #     images_att = net_attack(images,labels, attack)
            
        #    # print(images_att)
            
        #    # print(images_att-images)
            
        #     #print('processing image:' images)
        #     #model = model.to(device)
        #     #print(model.device)

        #     optimizer.zero_grad()
        #     prediction = model(images_att)
            
        #     #print('prediction value is :', prediction )

        #     loss = CE_loss(prediction, labels)
        #     # print('loss value is :', loss )
        #     loss.backward()

        #     optimizer.step()

        # correct_1 = 0
        
        # correct_2 = 0
        
        # correct_3 = 0
        
        #if i%10 == 0:
            #a = 0

        model.eval()
        correct_final_nat = 0
        correct_final_l2 = 0
        correct_final_linf = 0

        
        
        acc_nat, best_acc_nat, acc_l2, best_acc_l2, acc_linf, best_acc_linf, best_acc_aver = val(val_loader, device, model,  basic_model, AttackPGD, config_l2, config_linf, attack,\
        correct_final_nat, best_acc_nat, correct_final_l2, best_acc_l2, correct_final_linf,\
            best_acc_linf, best_acc_aver, args.checkpoint_loc)
        
        #acc_1, best_acc_nat = val_clean(val_loader, device, model, correct_final_1, best_acc_nat, args.checkpoint_loc)
        
        print('Epoch: ', i+1, ' Done!!  Natural  Accuracy: ', acc_nat)
        print('Epoch: ', i+1, '  Best Natural  Accuracy: ', best_acc_nat) 
          
        
        
        
        #acc_2, best_acc_l2 = val_adv(val_loader, device, model,  model, AttackPGD, config_l2, attack, correct_final_2, best_acc_l2 ,args.checkpoint_loc)
        
        print('Epoch: ', i+1, ' Done!!  l2(50, ..., 110)  Accuracy: ', acc_l2)
        print('Epoch: ', i+1, '  Best l2  Accuracy: ', best_acc_l2)         
        
        #acc_3, best_acc_linf = val_adv(val_loader, device, model,  basic_model, AttackPGD, config_linf, attack, correct_final_3, best_acc_linf ,args.checkpoint_loc)


        
        print('Epoch: ', i+1, ' Done!!  l_inf(5, ..., 11)  Accuracy: ', acc_linf)
        print('Epoch: ', i+1, '  Best l_inf  Accuracy: ', best_acc_linf)
        
        #print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

        print('Epoch: ', i+1, ' Done!!  average Accuracy: ', (acc_nat*2 + sum(acc_l2) + sum(acc_linf))/6 )
        print('Epoch: ', i+1, '  Best average  Accuracy: ', best_acc_aver)

    
        # for images, labels in val_loader:
            
        #     #a = a+1
        #     net_attack = AttackPGD(basic_model,config_linf)
            
        #     #print(net_attack)
            
        #     net_attack = net_attack.to(device)
        #     #print('processing testing image:', a)
        #     #images, labels = utils.cuda([images, labels], args.gpu_ids)
        #     images = images.to(device)
        #     labels = labels.to(device)
            
        #     images_att = net_attack(images,labels, attack)

        #     prediction = model(images_att)
        #     pred = prediction.argmax(dim=1, keepdim=True)
        #     correct_3 = pred.eq(labels.view_as(pred)).sum().item()
        #     #correct_final.append(correct)
        #     correct_final_3 = correct_final_3 + correct_3
        
        # acc = correct_final_3 / len(val_loader.dataset)
    
    #torch.save(model.state_dict(), args.checkpoint_loc)
            

