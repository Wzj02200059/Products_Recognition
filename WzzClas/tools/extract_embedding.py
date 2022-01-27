import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from copy import deepcopy
import importlib
import logging

import argparse
from collections import defaultdict
from tqdm import tqdm
import sys

sys.path.append("../")

from zzClassifier.datasets import Product_Dataloader_Close, Product_Dataloader_Open
from zzClassifier.losses import rzloss, FocalLoss
from zzClassifier.models import gan
from zzClassifier.models.resnet import resnet18
from zzClassifier.models.metrics import ArcMarginProduct
from zzClassifier.models.models import classifier32, classifier32ABN

from zzClassifier.core import train, train_cs, test, build_optimizer
from zzClassifier.core.model_builder import build_model

from utils import Logger, save_networks, load_networks
from tools.validator import Pairs_Validator
from tools.trainer import Pairs_Trainer


# Dataset
def get_args():
    parser = argparse.ArgumentParser("Training")

    # dataset
    parser.add_argument('--dataset', type=str, default='product')
    parser.add_argument('--train_dataroot', type=str, default='/home/ppz/wzj/cac-openset-master/data/jml/train')
    parser.add_argument('--val_dataroot', type=str, default='/home/ppz/wzj/cac-openset-master/data/jml/val')
    parser.add_argument('--unknown_train_dataroot', type=str,
                        default='/home/ppz/wzj/cac-openset-master/data/jml/unknow_train')
    parser.add_argument('--unknown_val_dataroot', type=str,
                        default='/home/ppz/wzj/cac-openset-master/data/jml/unknow_val')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')

    # optimization
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.5, help="threshold value to determin unseen")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
    parser.add_argument('--optimizer', type=str, default='SGD', help="optimizer")
    parser.add_argument('--decay_step', type=float, default=2, help="LEARNING_DECAY_STEP")
    parser.add_argument('--decay_gamma', type=float, default=0.9, help="LEARNING_DEACAY_GAMMA")
    parser.add_argument('--margin', type=float, default=0.25, help="margin")
    parser.add_argument('--gamma', type=float, default=80, help="gamma")
    parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
    parser.add_argument('--epoch', type=int, default=100)

    # model
    parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
    parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
    parser.add_argument('--model_name', type=str, default='RejectModel')
    parser.add_argument('--metrics', type=str, default='arc_margin')
    parser.add_argument('--backbone', type=str, default='resnet18')

    # misc
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--ns', type=int, default=1)
    parser.add_argument('--eval-freq', type=int, default=1)
    parser.add_argument('--eval_period', type=int, default=2)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-cpu', action='store_true')
    parser.add_argument('--save-dir', type=str, default='../log')
    parser.add_argument('--loss', type=str, default='rzloss')
    parser.add_argument('--eval', action='store_true', help="Eval", default=False)
    parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    options = vars(args)

    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # load dataset
    print("{} Preparation".format(options['dataset']))

    Data = Product_Dataloader_Open(train_dataroot=options['train_dataroot'], val_dataroot=options['val_dataroot'],
                                   unknown_train_dataroot=options['unknown_train_dataroot'],
                                   unknown_val_dataroot=options['unknown_val_dataroot'],
                                   batch_size=options['batch_size'], img_size=options['img_size'])
    trainloader, testloader, unknow_trainloader, unknow_valloader = Data.train_loader, Data.test_loader, Data.unknow_train_loader, Data.unknow_val_loader

    options['num_classes'] = Data.num_classes

    # initial model
    print("Creating model: {}".format(options['model_name']))
    net = build_model(options)

    if options['metrics'] == 'arc_margin':
        metric_fc = ArcMarginProduct(512, options['num_classes'], s=30, m=0.5, easy_margin=False)



if __name__ == '__main__':
    main()