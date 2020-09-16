

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
from weather_generation import *
from image_trans import *
import matplotlib.pyplot as plt
from PIL import Image


def train_clean (train_loader, device,optimizer,model,CE_loss):
    j = 0
    for j in range(1,3):    
        print('Doing clean images training No. ' + str(j))
        for images, labels in tqdm(train_loader):
                #j = j + 1
                
                #print('processing image No.:', j)
                #images, labels = utils.cuda([images, labels], args.gpu_ids)
                #images, labels = images.to(device), labels.to(device)
                
                #print(images)
                
                #net_attack = AttackPGD(basic_model,config_linf)
                
                #print(net_attack)
                
                #net_attack = net_attack.to(device)
                images = images.to(device)
                labels = labels.to(device)
                #print(images.device)
                # images.cuda(args.gpu_ids[0])
                # labels.cuda(args.gpu_ids[0])
                #print(images)
                #print(images.shape)
                #print(type(images))
                
                #images_att = net_attack(images,labels, attack)
                
               # print(images_att)
                
               # print(images_att-images)
                
                #print('processing image:' images)
                #model = model.to(device)
                #print(model.device)
    
                optimizer.zero_grad()
                prediction, weights = model(images)
                
                #print('prediction value is :', prediction )
    
                loss = CE_loss(prediction, labels)
                # print('loss value is :', loss )
                loss.backward()
    
                optimizer.step()

    return weights

def train_fog (train_loader, device,optimizer,model,CE_loss, config_fog):
    
    print('training snow step')
    
    j = 0
    
    for i in config_fog:
        j = j+1 
        
        b = 0
        
        print('fog Training ' + str(config_fog[i]['t']) + ' t:' + str(j))

        for images, labels in tqdm(train_loader):
            
            b = b+1
            
            #print('training ' + str(b) + ' batch')    
    
           
    
            images = images.to(device)
            labels = labels.to(device)
            
            for a in range(0,images.shape[0]):
                    
                   # print(images[i])
                    images_fog = add_fog(images[a], config_fog[i]['t'], config_fog[i]['light'])
                    #print(images_fog)
                    images[a] = images_fog 
    # add fog to images
    
    
            optimizer.zero_grad()
            prediction, weights_fog = model(images)
            
            #print('prediction value is :', prediction )
    
            loss = CE_loss(prediction, labels)
            # print('loss value is :', loss )
            loss.backward()
    
            optimizer.step()


    return weights_fog

def train_snow (train_loader, device,optimizer,model,CE_loss, config_snow):
    
    print('training snow step')
    
    j = 0

    for i in config_snow:

        j = j+1
      #  print('snow Training ' + str(config_snow[i]) + ' No.' + str(j))    
        
        b = 0
        
        for images, labels in tqdm(train_loader):
            
            b = b+1
            
           # print('training ' + str(b) + ' batch')    
                  
    
            images = images.to(device)
            labels = labels.to(device)
            
            for a in range(0,images.shape[0]):
                    
                   # print(images[i])
                    images_snow = add_snow(images[a], config_snow[i])
                    #print(images_fog)
                    images[a] = images_snow
    # add fog to images
    
    
            optimizer.zero_grad()
            prediction, weights_snow = model(images)
            
            #print('prediction value is :', prediction )
    
            loss = CE_loss(prediction, labels)
            # print('loss value is :', loss )
            loss.backward()
    
            optimizer.step()

    return weights_snow

def val(val_loader, device, model,  basic_model,\
        correct_final_nat, best_acc_nat, correct_final_fog, \
            best_acc_fog, correct_final_snow, best_acc_snow, checkpoint_loc, \
                config_fog, config_snow, best_acc_aver):
    
    print('testing clean step')
    
    acc_fog = []
    acc_snow = []

    
    for images_1, labels in val_loader:
            
 
            images_1 = images_1.to(device)
            labels = labels.to(device)
            
            #images_att = net_attack(images,labels, attack)

                  
            prediction, weights_clean = model(images_1)
            pred = prediction.argmax(dim=1, keepdim=True)
            correct_1 = pred.eq(labels.view_as(pred)).sum().item()
            #correct_final.append(correct)
            correct_final_nat = correct_final_nat + correct_1
            
    acc_nat = correct_final_nat / len(val_loader.dataset)
    
    print('testing fog step')
    
    j = 0
    
    for i in config_fog:
        
        j = j+1

        print('Valuation fog No. ' + str(j) ) 
        correct_2 = 0
        correct_final_fog = 0
        
        for images_2, labels in tqdm(val_loader):
                
                #a = a+1
                #net_attack = AttackPGD(basic_model,config_l2)
               
                #print(net_attack)
                
                #net_attack = net_attack.to(device)
                #print('processing testing image:', a)
                #images, labels = utils.cuda([images, labels], args.gpu_ids)
                images_2 = images_2.to(device)
                labels = labels.to(device)
                
                for a in range(0,images_2.shape[0]):
                     # print(images[i])
                      images_fog = add_fog(images_2[a], config_fog[i]['t'], config_fog[i]['light'])
                      #print(images_fog)
                      images_2[a] = images_fog 
                
                #images_att = net_attack(images,labels, attack)
    
    
    
                prediction, weights_fog = model(images_2)
                pred = prediction.argmax(dim=1, keepdim=True)
                correct_2 = pred.eq(labels.view_as(pred)).sum().item()
                #correct_final.append(correct)
                correct_final_fog = correct_final_fog + correct_2
        
        acc_fog.append(correct_final_fog / len(val_loader.dataset))

    print('testing snow step')

    j = 0
    
    for i in config_snow:
        
        j = j+1
        
        print('Valuation snow No. ' + str(j) )        
        correct_3 = 0
        correct_final_snow = 0
    
        for images_3, labels in tqdm(val_loader):
                
                #a = a+1
                # net_attack = AttackPGD(basic_model,config_linf)
                
                # #print(net_attack)
               
                # net_attack = net_attack.to(device)
                #print('processing testing image:', a)
                #images, labels = utils.cuda([images, labels], args.gpu_ids)
                images_3 = images_3.to(device)
                labels = labels.to(device)
                
                for a in range(0,images_3.shape[0]):
                    
                      
                    
                     # print(images[i])
                      images_snow = add_snow(images_3[a], config_snow[i])
                      #print(images_fog)
                      images_3[a] = images_snow 
                
               # images_att = net_attack(images,labels, attack)
    
                prediction, weights_snow = model(images_3)
                pred = prediction.argmax(dim=1, keepdim=True)
                correct_3 = pred.eq(labels.view_as(pred)).sum().item()
                #correct_final.append(correct)
                correct_final_snow = correct_final_snow + correct_3
            
        acc_snow.append(correct_final_snow / len(val_loader.dataset))    
    
    
    
    
    if (acc_nat*2 + sum(acc_fog) + sum(acc_snow))/6 > best_acc_aver:
            print('saving..')
            
            state = {
            'net': model.state_dict(),
            'acc_clean': acc_nat,
            'acc_fog': acc_fog,
            'acc_snow': acc_snow,
            #'epoch': epoch,
            }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, checkpoint_loc)
            best_acc_nat = acc_nat
            best_acc_fog = acc_fog
            best_acc_snow = acc_snow
            best_acc_aver = (best_acc_nat*2 + sum(best_acc_fog) + sum(best_acc_snow))/6
    
    return acc_nat, best_acc_nat, acc_fog, best_acc_fog, acc_snow, best_acc_snow, best_acc_aver






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
    

    config_fog_1 ={
        't': 0.12,
        'light': 0.8
        }

    config_fog_2 ={
        't': 0.13,
        'light': 0.6
        }
    
    

  
    config_fog = dict(config_fog_1 = config_fog_1, config_fog_2 = config_fog_2)
    
        
    brightness_1 = 2.0
    brightness_2 = 2.5
    
    config_snow = dict(brightness_1 = brightness_1, brightness_2 = brightness_2)
    
    attack = 'true'

    if args.dataset == 'cifar':
        output_classes = 10
    if args.dataset == 'tinyimagenet':
        output_classes = 200 
        
    global best_acc_nat, best_acc_l2, best_acc_linf
    best_acc_nat = 0
    best_acc_l2 = []
    best_acc_linf = []
    best_acc_fog = []
    best_acc_snow = []
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
    
    model_loc0 = './trained_model/ckptnat_resnet18_timagenet.pth'
    model_loc1 = './trained_model/ckptnat_resnet18_timagenet_fog_t_0.12_light_0.8.pth'
    model_loc2 = './trained_model/ckptnat_resnet18_timagenet_fog_t_0.13_light_0.6.pth'
    model_loc3 = './trained_model/ckptnat_resnet18_timagenet_snow_b_2.0.pth'
    model_loc4 = './trained_model/ckptnat_resnet18_timagenet_snow_b_2.5.pth'
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
    utils.load_model_nat(model_loc1, model.experts[1])
    utils.load_model_nat(model_loc2, model.experts[2])
    utils.load_model_nat(model_loc3, model.experts[3])
    utils.load_model_nat(model_loc4, model.experts[4])
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


    for i in range(args.epochs):
        model.train()
        #j = 0
        print('The epoch number is: ' + str(i))
        

        # train_clean (train_loader, device,optimizer,model,CE_loss)
        
        # # #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        # # #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        # train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_l2, attack)
                
        # train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_linf, attack)

        weights_nat = train_clean (train_loader, device,optimizer,model,CE_loss)
        
        print('nat training weights', weights_nat)

        # weights_linf = train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_linf, attack)
        
        # print('after linf training weights(final):', weights_linf) 

        #weights_linf = train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_linf, attack)
        
        weights_fog = train_fog (train_loader, device,optimizer,model,CE_loss, config_fog)
        
        print('after fog training weights(final):', weights_fog)         

        
        #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        #train_clean (train_loader, device,optimizer,model,CE_loss)
        
        # weights_l2 = train_adv(train_loader, device, optimizer, basic_model, model, AttackPGD ,CE_loss,config_l2, attack)
        
        # print('l2 training weights', weights_l2)     


        
        weights_snow = train_snow (train_loader, device,optimizer,model,CE_loss, config_snow)

        
        print('snow training weights', weights_snow)           
       

       

        model.eval()
        # correct_final_nat = 0
        # correct_final_l2 = 0
        # correct_final_linf = 0
        
        correct_final_nat = 0
        correct_final_fog = 0
        correct_final_snow = 0

        
        
        # acc_nat, best_acc_nat, acc_l2, best_acc_l2, acc_linf, best_acc_linf, best_acc_aver = val(val_loader, device, model,  basic_model, AttackPGD, config_l2, config_linf, attack,\
        # correct_final_nat, best_acc_nat, correct_final_l2, best_acc_l2, correct_final_linf,\
        #     best_acc_linf, best_acc_aver, args.checkpoint_loc)

        acc_nat, best_acc_nat, acc_fog, best_acc_fog, acc_snow, best_acc_snow, best_acc_aver = val(val_loader, device, model,  model,\
        correct_final_nat, best_acc_nat, correct_final_fog, best_acc_fog, correct_final_snow,\
            best_acc_snow, args.checkpoint_loc, config_fog, config_snow, best_acc_aver)
        
        #acc_1, best_acc_nat = val_clean(val_loader, device, model, correct_final_1, best_acc_nat, args.checkpoint_loc)
        
        print('Epoch: ', i+1, ' Done!!  Natural  Accuracy: ', acc_nat)
        print('Epoch: ', i+1, '  Best Natural  Accuracy: ', best_acc_nat) 
          
        

        print('Epoch: ', i+1, ' Done!!  fog  Accuracy: ', acc_fog)
        print('Epoch: ', i+1, '  Best fog  Accuracy: ', best_acc_fog)         
        


        
        print('Epoch: ', i+1, ' Done!!  snow  Accuracy: ', acc_snow)
        print('Epoch: ', i+1, '  Best snow  Accuracy: ', best_acc_snow)
        
        #print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

        print('Epoch: ', i+1, ' Done!!  average Accuracy: ', (acc_nat*2 + sum(acc_fog) + sum(acc_snow))/6 )
        print('Epoch: ', i+1, '  Best average  Accuracy: ', best_acc_aver)
        
        
      
        
        # print('Epoch: ', i+1, ' Done!!  l2(50, ..., 110)  Accuracy: ', acc_l2)
        # print('Epoch: ', i+1, '  Best l2  Accuracy: ', best_acc_l2)         
        


        
        # print('Epoch: ', i+1, ' Done!!  l_inf(5, ..., 11)  Accuracy: ', acc_linf)
        # print('Epoch: ', i+1, '  Best l_inf  Accuracy: ', best_acc_linf)
        
        # #print('Epoch: ', i+1, ' Done!!    Loss: ', loss)

        # print('Epoch: ', i+1, ' Done!!  average Accuracy: ', (acc_nat*2 + sum(acc_l2) + sum(acc_linf))/6 )
        # print('Epoch: ', i+1, '  Best average  Accuracy: ', best_acc_aver)

    

# def train_adv(train_loader, device,optimizer, basic_model, model, AttackPGD ,CE_loss,config,attack):
#     j = 0
#     for i in config:
#         j = j+1
#         print('Adv Training ' + str(config[i]['_type']) + '  epsilon:' + str(j))
        
#         for images_adv, labels in train_loader:
#                 #j = j + 1
                
#                 #print('processing image No.:', j)
#                 #images, labels = utils.cuda([images, labels], args.gpu_ids)
#                 #images, labels = images.to(device), labels.to(device)
                
#                 #print(images)
                
#                 net_attack = AttackPGD(basic_model,config[i])
                
#                 #print(net_attack)
                
#                 net_attack = net_attack.to(device)
#                 images_adv = images_adv.to(device)
#                 labels = labels.to(device)
#                 #print(images.device)
#                 # images.cuda(args.gpu_ids[0])
#                 # labels.cuda(args.gpu_ids[0])
#                 #print(images)
#                 #print(images.shape)
#                 #print(type(images))
                
#                 images_att = net_attack(images_adv,labels, attack)
                
#                # print(images_att)
                
#                # print(images_att-images)
                
#                 #print('processing image:' images)
#                 #model = model.to(device)
#                 #print(model.device)
    
#                 optimizer.zero_grad()
#                 prediction, weights = model(images_att)
                
#                 #print('prediction value is :', prediction )
    
#                 loss = CE_loss(prediction, labels)
#                 # print('loss value is :', loss )
#                 loss.backward()
    
#                 optimizer.step()

#     return weights
# def val(val_loader, device, model,  basic_model, AttackPGD, config_l2, config_linf, attack,\
#         correct_final_nat, best_acc_nat, correct_final_l2, best_acc_l2, correct_final_linf,\
#             best_acc_linf, best_acc_aver, checkpoint_loc):
    
#     acc_linf = []
#     acc_l2 = []
    
#     print('Valuation clean images')
    
#     for images_1, labels in val_loader:
            
 
#             images_1 = images_1.to(device)
#             labels = labels.to(device)
            
#             #images_att = net_attack(images,labels, attack)
    
#             prediction, weights_nat = model(images_1)
#             pred = prediction.argmax(dim=1, keepdim=True)
#             correct_1 = pred.eq(labels.view_as(pred)).sum().item()
#             #correct_final.append(correct)
#             correct_final_nat = correct_final_nat + correct_1
            
#     acc_nat = correct_final_nat / len(val_loader.dataset)
    
#     j=0
    
#     for i in config_l2:
#         j = j + 1
#         #print('Valuation l_2  epsilon: ' + str(config_l2[i]['epsilon']))
#         print('Valuation l_2  epsilon: 50+ ' + str(j) + '*10 / 255')
        
#         correct_2 = 0
#         correct_final_l2 = 0
        
#         for images_2, labels in val_loader:
                
#                 #a = a+1
#                 net_attack = AttackPGD(basic_model,config_l2[i])
                
#                 #print(net_attack)
                
#                 net_attack = net_attack.to(device)
#                 #print('processing testing image:', a)
#                 #images, labels = utils.cuda([images, labels], args.gpu_ids)
#                 images_2 = images_2.to(device)
#                 labels = labels.to(device)
                
#                 images_att = net_attack(images_2,labels, attack)
    
#                 prediction,weights_l2 = model(images_att)
#                 pred = prediction.argmax(dim=1, keepdim=True)
#                 correct_2 = pred.eq(labels.view_as(pred)).sum().item()
#                 #correct_final.append(correct)
#                 correct_final_l2 = correct_final_l2 + correct_2
            
#         acc_l2.append(correct_final_l2 / len(val_loader.dataset))
    
#     j = 0
#     for i in config_linf:
        
#         j = j+1
    
#         #print('Valuation' + str(config_linf[i]['_type']) + '  epsilon:' + str(config_linf[i]['epsilon']))
#         print('Valuation l_inf  epsilon: 5+ ' + str(j) + '*10 / 255')
#         correct_2 = 0
#         correct_final_linf = 0
#         for images_3, labels in val_loader:
                
#                 #a = a+1
#                 net_attack = AttackPGD(basic_model,config_linf[i])
                
#                 #print(net_attack)
                
#                 net_attack = net_attack.to(device)
#                 #print('processing testing image:', a)
#                 #images, labels = utils.cuda([images, labels], args.gpu_ids)
#                 images_3 = images_3.to(device)
#                 labels = labels.to(device)
                
#                 images_att = net_attack(images_3,labels, attack)
    
#                 prediction, weights_linf = model(images_att)
#                 pred = prediction.argmax(dim=1, keepdim=True)
#                 correct_3 = pred.eq(labels.view_as(pred)).sum().item()
#                 #correct_final.append(correct)
#                 correct_final_linf = correct_final_linf + correct_3
            
#         acc_linf.append(correct_final_linf / len(val_loader.dataset))    
    
    
    
    
#     if (acc_nat*2 + sum(acc_l2) + sum(acc_linf))/6 > best_acc_aver:
#             print('saving..')
            
#             state = {
#             'net': model.state_dict(),
#             'acc_clean': acc_nat,
#             'acc_l2': acc_l2,
#             'acc_linf': acc_linf,
#             #'epoch': epoch,
#             }

#             if not os.path.isdir('checkpoint'):
#                 os.mkdir('checkpoint')
#             torch.save(state, checkpoint_loc)
#             best_acc_nat = acc_nat
#             best_acc_l2 = acc_l2
#             best_acc_linf = acc_linf
#             best_acc_aver = (best_acc_nat*2 + sum(best_acc_l2) + sum(best_acc_linf))/6
    
#     return acc_nat, best_acc_nat, acc_l2, best_acc_l2, acc_linf, best_acc_linf, best_acc_aver