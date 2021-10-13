import argparse
import time
import datetime
import os
import shutil
import sys
import random

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.loss import get_segmentation_loss
from core.utils.distributed import *
from core.utils.logger import setup_logger
from core.utils.lr_scheduler import WarmupPolyLR
from core.utils.score import SegmentationMetric,hist_info, compute_score
# from core.utils.AutomaticWeightedLoss import AutomaticWeightedLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default='fdsnet',
                        choices=['fcn32s', 'fcn16s', 'fcn8s',
                                 'fcn', 'psp', 'deeplabv3', 'deeplabv3_plus',
                                 'danet', 'denseaspp', 'bisenet',
                                 'encnet', 'dunet', 'icnet',
                                 'enet', 'ocnet', 'ccnet', 'psanet',
                                 'cgnet', 'espnet', 'lednet', 'dfanet','fastscnn','u2netp',
                                 'emanet','afn','deeplabv3plus','fdsnet','fdsnet0','bisenetv2'],
                        help='model name (default: fcn32s)')
    parser.add_argument('--backbone', type=str, default='',
                        choices=['vgg16', 'resnet18', 'resnet50',
                                 'resnet101', 'resnet152', 'densenet121',
                                 'densenet161', 'densenet169', 'densenet201','xception'],
                        help='backbone name (default: vgg16)')
    parser.add_argument('--dataset', type=str, default='phone_voc',
                        choices=['pascal_voc', 'pascal_aug', 'ade20k',
                                 'citys', 'sbu','mt_voc','phone_voc','sd_voc','aitex_voc'],
                        help='dataset name (default: pascal_voc)')
    parser.add_argument('--base-size', type=int, default=None,#520
                        help='base image short width size')
    parser.add_argument('--crop-size', type=int, default=None,# 450
                        help='crop image size')
    parser.add_argument('--pad-size', type=int, default=None,# data padding
                        help='padding image size')
    parser.add_argument('--scale-ratio', type=float, default=0.75,# image scale
                        help='scale image size')

    parser.add_argument('--workers', '-j', type=int, default=4,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--jpu', action='store_true', default=False,
                        help='JPU')
    parser.add_argument('--use-ohem', type=bool, default=True,
                        help='OHEM Loss for cityscapes dataset')
    parser.add_argument('--use-focal', type=bool, default=False,
                        help='Focal Loss') 
    parser.add_argument('--pretrained', type=bool, default=False, 
                        help='use pretrained model pth') 

    parser.add_argument('--aux', type=bool,default=True, # use the auxiliary tasks
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5, 
                        help='auxiliary loss weight')
    parser.add_argument('--aux-weight2', type=float, default=0.5,  
                        help='auxiliary semantic loss weight')    
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N', 
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--seed', type=int, default=2, metavar='N',  
                        help='random seed')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--warmup-iters', type=int, default=0,
                        help='warmup iters')
    parser.add_argument('--warmup-factor', type=float, default=1.0 / 3,
                        help='lr = warmup_factor * lr')
    parser.add_argument('--warmup-method', type=str, default='linear',
                        help='method of warmup')
    # cuda setting
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--local_rank', type=int, default=0)
    # checkpoint and log
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--save-dir', default='/data/zhangj/.torch/models',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--save-epoch', type=int, default=10,
                        help='save model every checkpoint-epoch')
    parser.add_argument('--log-dir', default='/data/zhangj/runs/logs/',
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='print log every log-iter')
    # evaluation only
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='run validation every val-epoch')
    parser.add_argument('--skip-val', action='store_true', default=False,
                        help='skip validation during training')
    args = parser.parse_args()

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'pascal_aug': 80,
            'pascal_voc': 50,
            'pcontext': 80,
            'ade20k': 160,
            'citys': 120,
            'sbu': 160,
            'mt_voc': 50,
            'phone_voc': 50,
        }
        args.epochs = epoches[args.dataset.lower()]
    if args.lr is None:
        lrs = {
            'coco': 0.004,
            'pascal_aug': 0.001,
            'pascal_voc': 0.0001,
            'pcontext': 0.001,
            'ade20k': 0.01,
            'citys': 0.01,
            'sbu': 0.001,
            'mt_voc': 0.0001,
            'phone_voc': 0.0001,
        }
        args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 
    np.random.seed(seed)
    torch.manual_seed(seed) #
    torch.cuda.manual_seed(seed)#
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    np.random.seed(int(args.seed)+worker_id)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform 
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 
                        'crop_size': args.crop_size, 'pad_size': args.pad_size, 'scale_ratio':args.scale_ratio}
        train_dataset = get_segmentation_dataset(args.dataset, split='train', mode='train', **data_kwargs)
        val_dataset = get_segmentation_dataset(args.dataset, split='test', mode='val', **data_kwargs)
        args.iters_per_epoch = len(train_dataset) // (args.num_gpus * args.batch_size)
        args.max_iters = args.epochs * args.iters_per_epoch

        # train_sampler = make_data_sampler(train_dataset, shuffle=True, distributed=args.distributed)
        # train_batch_sampler = make_batch_data_sampler(train_sampler, args.batch_size, args.max_iters)
        # val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        # val_batch_sampler = make_batch_data_sampler(val_sampler, 1) #args.batch_size

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            # batch_sampler=train_batch_sampler,
                                            batch_size=args.batch_size,
                                            shuffle=True,
                                            num_workers=args.workers,
                                            pin_memory=True,
                                            worker_init_fn=seed_torch)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          # batch_sampler=val_batch_sampler,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=args.workers,
                                          pin_memory=True,
                                          worker_init_fn=seed_torch)

        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,root=args.save_dir,
                                            pretrained=args.pretrained, #local_rank=args.local_rank,
                                            aux=args.aux, jpu=args.jpu, norm_layer=BatchNorm2d).to(self.device)

        # resume checkpoint if needed
        if args.resume:
            if os.path.isfile(args.resume):
                name, ext = os.path.splitext(args.resume)
                assert ext == '.pkl' or '.pth', 'Sorry only .pth and .pkl files supported.'
                print('Resuming training, loading {}...'.format(args.resume))
                self.model.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))

        # create criterion
        self.criterion = get_segmentation_loss(args.model, use_ohem=args.use_ohem,use_focal=args.use_focal, aux=args.aux,
                                               aux_weight=args.aux_weight, ignore_index=-1).to(self.device)
        # self.awl = AutomaticWeightedLoss(3) # 自动调整权重

        # optimizer, for model just includes pretrained, head and auxlayer
        params_list = list()
        if hasattr(self.model, 'pretrained'):
            params_list.append({'params': self.model.pretrained.parameters(), 'lr': args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params_list.append({'params': getattr(self.model, module).parameters(), 'lr': args.lr * 10})
        # self.optimizer = torch.optim.SGD(params_list,
        #                                  lr=args.lr,
        #                                  momentum=args.momentum,
        #                                  weight_decay=args.weight_decay)

        # params_list.append({'params': self.awl.parameters(), 'weight_decay': 0, 'lr':1e-3})
        self.optimizer=torch.optim.Adam(params_list, weight_decay=10e-6, lr=args.lr)

        # lr scheduling
        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                         max_iters=args.max_iters,
                                         power=0.9,
                                         warmup_factor=args.warmup_factor,
                                         warmup_iters=args.warmup_iters,
                                         warmup_method=args.warmup_method)

        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.local_rank],
                                                             output_device=args.local_rank)

        # evaluation metrics
        self.metric = SegmentationMetric(train_dataset.num_class)

        self.best_pred = 0.0
        self.num_cls = val_dataset.NUM_CLASS
        self.label_name =val_dataset.classes

    def train(self):
        save_to_disk = get_rank() == 0
        epochs, max_iters = self.args.epochs, self.args.max_iters
        log_per_iters, val_per_iters = self.args.log_iter, self.args.val_epoch * self.args.iters_per_epoch
        save_per_iters = self.args.save_epoch * self.args.iters_per_epoch
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))

        self.model.train()
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(self.args.epochs):
            loss_epoch = 0
            for iteration, (images, targets,tg_edge,tg_label,name) in enumerate(self.train_loader):
            # for iteration, (images, targets,name) in enumerate(self.train_loader):
                iteration = iteration + 1+epoch*self.args.iters_per_epoch

                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                
                loss_dict = 0
                if args.aux and (args.model=='fdsnet'):
                    tg_edge.to(self.device)
                    tg_label.to(self.device)
                    loss_dict = self.criterion(outputs,(targets,tg_edge,tg_label))
                else:
                    loss_dict = self.criterion(outputs, targets)
                # loss_dict = self.criterion(outputs, targets)
                loss_weight = [1.0, args.aux_weight, args.aux_weight2]
                losses = sum(loss*weight for loss,weight in zip(loss_dict.values(),loss_weight))
                # losses_tuple = tuple(loss_dict.values())
                # losses =self.awl(*losses_tuple) # 
                loss_epoch += losses

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                self.optimizer.zero_grad()
                losses.backward()

                self.optimizer.step()
                self.lr_scheduler.step()


                if iteration % log_per_iters == 0 and save_to_disk:
                    logger.info(
                        "Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} ".format(
                            iteration, max_iters, self.optimizer.param_groups[0]['lr'], losses_reduced.item(),))

                if iteration % save_per_iters == 0 and save_to_disk:
                    save_checkpoint(self.model, self.args, is_best=False)

                if not self.args.skip_val and iteration % val_per_iters == 0:
                    self.validation()
                    self.model.train()

            eta_seconds = ((time.time() - start_time) / (epoch+1e-5)) * (self.args.epochs - epoch)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info("Epoch: {:d}/{:d}  || Loss: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                            epoch, self.args.epochs, loss_epoch/self.args.iters_per_epoch,
                            str(datetime.timedelta(seconds=int(time.time() - start_time))), eta_string))

        save_checkpoint(self.model, self.args, is_best=False)
        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
                total_training_str, total_training_time / max_iters))

    def validation(self):
        # total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        is_best = False
        self.metric.reset()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        torch.cuda.empty_cache()  # TODO check if it helps
        model.eval()
        hist_all = 0
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device) # 
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image)
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            pred = torch.argmax(outputs[0], 1)
            # pred = pred.cpu().data.numpy()
            hist = hist_info(target,pred,self.num_cls)
            hist_all += hist
            # logger.info("Sample: {:d}, Validation pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc, mIoU))

        scores, cls_iu = compute_score(hist_all.cpu(), self.label_name)
        pixAccA, mIoU = scores['pAcc'], scores['mIoU']
        logger.info("!!All Validation pixAcc: {:.4f}, mIoU: {:.4f}, Class IoU: {}".format(pixAcc, mIoU, cls_iu))
        new_pred = mIoU # (pixAcc + mIoU) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        save_checkpoint(self.model, self.args, is_best) 
        synchronize()


def save_checkpoint(model, args, is_best=False):
    """Save Checkpoint"""
    directory = os.path.expanduser(args.save_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}.pth'.format(args.model, args.backbone, args.dataset)
    filename = os.path.join(directory, filename)

    if args.distributed:
        model = model.module
    torch.save(model.state_dict(), filename)
    if is_best:
        best_filename = '{}_{}_{}_best_model.pth'.format(args.model, args.backbone, args.dataset)
        best_filename = os.path.join(directory, best_filename)
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    args = parse_args()
    args.seed =  int(time.time() )
    seed_torch(args.seed)

    # reference maskrcnn-benchmark
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.num_gpus = num_gpus
    args.distributed = num_gpus > 1
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
    else:
        args.distributed = False
        args.device = "cpu"
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()
    args.lr = args.lr * num_gpus

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(), filename='{}_{}_{}_log.txt'.format(
        args.model, args.backbone, args.dataset))
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    trainer = Trainer(args)
    trainer.train()
    torch.cuda.empty_cache()
