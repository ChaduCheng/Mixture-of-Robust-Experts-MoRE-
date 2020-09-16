import os
from argparse import ArgumentParser
#from testing import validation
#from trained_model_expert import train
from train_mod_adv import train
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
    args.batch_size = 100
    args.train_split = 0.8 
    args.lr = 0.001
    args.gpu_ids = '0'
    args.checkpoint_loc = './checkpoint/ckptMoE_resnet_timagenet_clean+adv.pth'
    #args.checkpoint_loc = './trained_model/ckptl2_alex_cifar_50.pth'
    args.num_experts = 5
    args.training = True
    args.testing = False
    # args.tr       aining = False
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