import os
import argparse
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as ddp

from zzClassifier.models import *
from zzClassifier.losses import rzloss, FocalLoss, SmoothAP, CrossEntropyLabelSmooth, TripletLoss
from zzClassifier.datasets import Triplet_Dataloader, Smooth_Dataloader, batch_sampler, BalancedBatchSampler
from zzClassifier.core import build_optimizer, build_model
from tools.utils import save_networks, load_pretrain, setup_logger, get_class_list
from tools.validator_embedding import retrieval_validation
from tools.trainer_embedding import Normal_Classifier_Trainer
from configs import classifier_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda")

# Dataset
def get_args():
    parser = argparse.ArgumentParser("Training")
    # dataset
    parser.add_argument('--classifier', type=str, default='metric classifier')

    parser.add_argument('--train_txt', type=str, default='pretrain/train.txt')
    parser.add_argument('--val_txt', type=str, default='pretrain/test.txt')
    parser.add_argument('--synset_txt', type=str, default='pretrain/synset.txt')
    parser.add_argument('--test_template', type=str, default='../dataset/TestImages/template_images')
    parser.add_argument('--test_query', type=str, default='../dataset/TestImages/real_subimages')
    parser.add_argument('--test_openset', type=str, default='../dataset/TestImages/openset')

    # optimization
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--threshold', type=float, default=0.6, help="threshold value to determin unseen")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate for model")
    parser.add_argument('--optimizer', type=str, default='SGD', help="optimizer")
    parser.add_argument('--decay_step', type=float, default=2, help="LEARNING_DECAY_STEP")
    parser.add_argument('--decay_gamma', type=float, default=0.8, help="LEARNING_DEACAY_GAMMA")
    parser.add_argument('--margin', type=float, default=1, help="margin")
    parser.add_argument('--gamma', type=float, default=80, help="gamma")
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--norm', type=float, default=2)
    parser.add_argument('--momentum', type=float, default=0.9)

    # model
    parser.add_argument('--model_name', type=str, default='RESNET_Model')
    parser.add_argument('--id_loss', type=str, default='arcface')
    parser.add_argument('--metrics', type=str, default='celoss')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--pretrained_model', type=str, default='work_space/out20210103/epoch_20_0.8550106609808102.pth')
    parser.add_argument('--pretrained_config', type=str, default='checkpoints/cls_config.yaml')
    parser.add_argument('--save_dir', type=str, default='work_space/out20210103')

    # misc
    parser.add_argument('--eval_period', type=int, default=10)
    parser.add_argument('--gpu', type=str, default='0,1,2,3')
    parser.add_argument("--local_rank", type=int,default=-1,help='DDP parameter, do not modify')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument("--resume", type=bool, default=True,
                        help="whether to attempt to resume from the checkpoint directory")
    parser.add_argument("--ddp", type=bool, default=True, help="chose to use ddp mode")
    args = parser.parse_args()
    return args


def update_config(cfg, args):
    class_list = get_class_list(args.synset_txt)
    num_class = len(class_list)

    cfg = deepcopy(cfg)

    cfg.dataset.train_txt = args.train_txt
    cfg.dataset.val_txt = args.val_txt 
    cfg.dataset.synset_txt = args.synset_txt
    cfg.dataset.test_template = args.test_template
    cfg.dataset.test_query = args.test_query
    cfg.dataset.test_openset = args.test_openset
    cfg.dataset.num_class = num_class
    cfg.dataset.img_size = args.img_size
    cfg.dataset.batch_size = args.batch_size

    cfg.model.model_name = args.model_name
    cfg.model.id_loss = args.id_loss
    cfg.model.metrics = args.metrics
    cfg.model.backbone = args.backbone
    cfg.model.save_dir = args.save_dir
    cfg.model.pretrained_model = args.pretrained_model
    cfg.model.pretrained_config = args.pretrained_config

    cfg.optimization.threshold = args.threshold
    cfg.optimization.lr = args.lr
    cfg.optimization.optimizer = args.optimizer
    cfg.optimization.decay_step = args.decay_step
    cfg.optimization.decay_gamma = args.decay_gamma
    cfg.optimization.margin = args.margin
    cfg.optimization.epoch = args.epoch
    cfg.optimization.norm = args.norm
    cfg.optimization.momentum = args.momentum

    cfg.misc.eval_period = args.eval_period
    cfg.misc.gpu = args.gpu
    cfg.misc.seed = args.seed
    cfg.misc.num_workers = args.num_workers
    cfg.misc.resume = args.resume
    cfg.misc.local_rank = args.local_rank
    cfg.misc.world_size = args.world_size
    cfg.misc.ddp = args.ddp

    return cfg, class_list


def main():
    args = get_args()
    # Setup the cfg: load the default params
    cfg = classifier_model._P

    # load resume pretrain model's params
    if args.resume:
        cfg.merge_from_file(args.pretrained_config)

    # Update the configs from argparse
    cfg, class_list = update_config(cfg, args) 

    # Params loading is finished, Save the config
    cfg.freeze() 
    with open(os.path.join(cfg.model.save_dir, 'cls_config.yaml'), 'w') as f:
        f.write(cfg.dump())

    # Setup gpu 
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.misc.gpu

    dist.init_process_group('nccl', "tcp://0.0.0.0:8064", world_size=cfg.misc.world_size, rank=cfg.misc.local_rank)
    world_size = dist.get_world_size()
    torch.cuda.set_device(cfg.misc.local_rank)
    process_group = torch.distributed.new_group()
    device = torch.device("cuda:{}".format(cfg.misc.local_rank))
    device = torch.device("cuda:{}".format(cfg.misc.gpu))

    # Setup logger
    logger = setup_logger(cfg.model.save_dir, name = cfg.model.logger_name)
    logger.info("~~~~~~~~~~~~~~START TRAINING~~~~~~~~~~~~~~")
    logger.info(args)
    logger.info("Config:\n" + str(cfg))

    # load dataset
    logger.info("Preparation Dataset from {}".format(cfg.dataset.train_txt))
    train_set = Smooth_Dataloader(cfg, class_list, status= 'train')

    # data_sampler = BalancedBatchSampler(train_set, n_classes = 32, n_samples = 4)
    # train_loader = DataLoader(train_set, num_workers=cfg.misc.num_workers, batch_sampler=data_sampler, pin_memory=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas = world_size, rank = cfg.misc.local_rank)
    train_loader = DataLoader(train_set, shuffle=(train_sampler is None), batch_size=cfg.dataset.batch_size, num_workers=cfg.misc.num_workers, \
                              drop_last=True, pin_memory=True, sampler=train_sampler)
    # val_set = Smooth_Dataloader(cfg, class_list, status= 'val')
    # val_loader = DataLoader(val_set, shuffle=True, batch_size=cfg.dataset.batch_size, num_workers=cfg.misc.num_workers, drop_last=True)

    # initial model
    logger.info("Creating model: {}".format(cfg.model.model_name))
    net = build_model(cfg)
    
    if dist.get_rank() == 0:
        logger.info('Model structure: \n' + str(net))
 
    # initial criterion
    assert cfg.model.metrics in ['rzloss', 'celoss', 'focal_loss', 'triplet_loss']
    if cfg.model.metrics == 'rzloss':
        criterion = rzloss(cfg.optimization.margin, cfg.optimization.gamma)
    elif cfg.model.metrics == 'celoss':
        criterion = nn.CrossEntropyLoss()
    elif cfg.model.metrics == 'focal_loss':
        criterion = FocalLoss(cfg.optimization.focal_gamma)
    elif cfg.model.metrics == 'triplet_loss':
        criterion = nn.TripletMarginLoss(margin=cfg.optimization.margin, p=cfg.optimization.norm)
    elif cfg.model.metrics == 'smoothap_loss':
        criterion = SmoothAP(anneal=0.01, batch_size=cfg.dataset.batch_size, num_id=6, feat_dims=256)

    # ce_criterion = CrossEntropyLabelSmooth(device, num_classes=cfg.dataset.num_class, use_gpu=True)
    ce_criterion = torch.nn.CrossEntropyLoss()
    # initial optimizer
    optimizer, scheduler = build_optimizer(cfg, net)

    # initial model trainer
    net = net.to(device)
    net = ddp(net, device_ids=[cfg.misc.local_rank])

    # if resume
    if cfg.misc.resume:
        net = load_pretrain(cfg, net)

    # if resume
    if cfg.misc.resume:
        net = load_pretrain(cfg, net)

    # if resume
    if cfg.misc.resume:
        net = load_pretrain(cfg, net)
        
    trainer = Normal_Classifier_Trainer(cfg, train_loader, device)
    # validator = Pairs_Validator(cfg, val_loader, device)

    # init amp 
    scaler = torch.cuda.amp.GradScaler()

    best_accuracy = 0
    logger.info("Ready for training model")

    for epoch in tqdm(range(cfg.optimization.epoch)):
        mb = epoch * 0.01
        criterion = TripletLoss(margin = 0.4 + mb)
        net = trainer.train_epoch(net, ce_criterion, criterion, optimizer, epoch, cfg, scaler)
        # Validation
        if epoch > 1:
            if epoch % cfg.misc.eval_period == 0:
                logger.info("Ready for validation model")
                # if dist.get_rank() == 0:
                close_accuracy, open_accuracy = retrieval_validation(cfg, net, device)
                # update the best validation accuracy
                logger.info('validaton in epoch {} , close acc is {}, open acc is {}'.format(epoch, close_accuracy, open_accuracy))
                if close_accuracy > best_accuracy:
                    best_accuracy = close_accuracy
                    # save the best model
                    save_networks(net, cfg, name=epoch, acc=best_accuracy)
                    logger.info('Save the best checkpoint weight file in path {}'.format(cfg.model.save_dir))
        scheduler.step()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
    