import os
from tqdm import tqdm
from argparse import ArgumentParser
#from testing import validation
from train import train    # do clean image training
# from train_l2 import train  # do l2 adv training
# from train_linf import train # do linf adv training
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
    args.dataset = 'tinyimagenet'
    args.batch_size = 64
    args.train_split = 0.8 
    args.lr = 0.001
    args.gpu_ids = '0'
    args.checkpoint_loc = './checkpoint/ckptnat_alexnet_timagenet.pth' # clean image
    #args.checkpoint_loc = './checkpoint/ckptl2_alexnet_timagenet_80_255.pth' # l_2 image
    #args.checkpoint_loc = './checkpoint/ckptlinf_alexnet_timagenet_8_255.pth' # l_inf image
    #args.checkpoint_loc = './checkpoint/ckptnat_resnet_timagenet_fog_t_0.13_l_0.6.pth' # fog image
    #args.checkpoint_loc = './checkpoint/ckptnat_resnet18_timagenet_snow_b_2.5.pth'  # snow image


    args.num_experts = 3
    args.training = True
    args.testing = False
    # args.training = False
    # args.testing = True
    return args

def main():
    args = get_args()
    # str_ids = args.gpu_ids.split(',')
    # args.gpu_ids = []
    # for str_id in str_ids:
    #     id = int(str_id)
    #     if id >= 0:
    #         args.gpu_ids.append(id)
    
    if args.training:
        train(args)
    if args.testing:
        test(args)


if __name__ == '__main__':
    main()