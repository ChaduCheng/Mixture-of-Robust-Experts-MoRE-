import os
from argparse import ArgumentParser
from train import train # do clean image training
# from train_l2 import train #do l2 training
# from train_linf import train # do linf training
# from train_fog import train # do fog training
# from train_snow import train # do snow training 
from tests import test


def get_args():
    parser = ArgumentParser(description='Mixture of Experts')
    # parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--dataset', type=float, default='mnist')
    # parser.add_argument('--batch_size', type=int, default=8)
    # parser.add_argument('--train_split', type=float, default=0.8)
    # parser.add_argument('--lr', type=float, default=.001)
    # parser.add_argument('--gpu_ids', type=str, default='0')
    # parser.add_argument('--checkpoint_loc', type=str, default='./checkpoint/latest_model.ckpt')
    # parser.add_argument('--num_experts', type=int, default=16)
    # parser.add_argument('--training', type=bool, default=True)
    # parser.add_argument('--testing', type=bool, default=False)
    args = parser.parse_args()
    args.epochs = 100
    args.dataset = 'cifar'
    args.batch_size = 100
    args.train_split = 0.8 
    args.lr = 0.1
    args.gpu_ids = '0'
    args.checkpoint_loc = './checkpoint/ckptnat_resnet18_cifar.pth' # clean model
    # args.checkpoint_loc = './checkpoint/ckptl2_resnet18_cifar.pth' # l2 adv training model
    # args.checkpoint_loc = './checkpoint/ckptlinf_resnet18_cifar.pth' #linf adv training model
    # args.checkpoint_loc = './checkpoint/ckptnat_resnet18_cifar_fog.pth' #fog training model
    # args.checkpoint_loc = './checkpoint/ckptnat_resnet18_cifar_snow_.pth' #snow training model
    args.num_experts = 3
    args.training = True
    args.testing = False
    # args.training = False
    # args.testing = True
    return args

def main():
    args = get_args()    
    if args.training:
        train(args)
    if args.testing:
        test(args)


if __name__ == '__main__':
    main()