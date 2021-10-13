from __future__ import print_function

import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import imgviz
import PIL.Image
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torchvision import transforms
from torchvision.transforms import ToPILImage
from core.data.dataloader import get_segmentation_dataset
from core.models.model_zoo import get_segmentation_model
from core.utils.score import SegmentationMetric,hist_info, compute_score
from core.utils.visualize import get_color_pallete
from core.utils.logger import setup_logger
from core.utils.distributed import synchronize, get_rank, make_data_sampler, make_batch_data_sampler

from train import parse_args
from torch2trt import torch2trt


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader 在eval时使用test数据集验证 test, testval
        val_dataset = get_segmentation_dataset(args.dataset, split='test', mode='val', transform=input_transform, #testval
                                                            scale_ratio=args.scale_ratio)
        val_sampler = make_data_sampler(val_dataset, False, args.distributed)
        val_batch_sampler = make_batch_data_sampler(val_sampler, images_per_batch=1) # images_per_batch=1
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_sampler=val_batch_sampler,
                                          num_workers=args.workers,
                                          pin_memory=True)
        self.d_loader = data.DataLoader(dataset=val_dataset,
                                        shuffle=False,
                                        batch_size=len(val_dataset))
        # create network
        BatchNorm2d = nn.SyncBatchNorm if args.distributed else nn.BatchNorm2d
        self.model = get_segmentation_model(model=args.model, dataset=args.dataset, backbone=args.backbone,
                                            aux=args.aux, pretrained=True, pretrained_base=False,
                                            local_rank=args.local_rank,
                                            norm_layer=BatchNorm2d).to(self.device)
        if args.distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model,
                                                             device_ids=[args.local_rank], output_device=args.local_rank)
        self.model.to(self.device)

        self.metric = SegmentationMetric(val_dataset.num_class)

        self.num_cls =val_dataset.NUM_CLASS
        self.label_name = val_dataset.classes
        self.root_path = val_dataset.voc_root
        self.hist = 0
    # 打印总的pa和miou，和分类别显示iou
    # 为什么要一张一张的计算，直接计算整个val的iou,问题是测试图像的长宽不一致
    def eval(self):
        # self.model.eval()
        # if self.args.distributed:
        #     model = self.model.module
        # else:
        #     model = self.model
        # images, targets, filenames = next(iter(self.d_loader))
        # with torch.no_grad():
        #     outputs = model(images)
        # hist, labeled, correct = hist_info(outputs, targets, self.num_cls)
        # iu, mean_IU, mean_IU_no_back, mean_pixel_acc = compute_score(hist,correct,labeled)
        # # logger.info("Average validation pixAcc: {:.3f}, all mIoU: {:.3f}".format(ave_pa * 100, ave_miou * 100))
        # print(iu, mean_IU, mean_IU_no_back, mean_pixel_acc)
        # 原来的代码
        self.metric.reset()
        self.model.eval()
        if self.args.distributed:
            model = self.model.module
        else:
            model = self.model
        # random_input = torch.ones([1, 3, 200, 200]).to(self.device)
        # random_input = torch.ones([1, 3, 810, 1440]).to(self.device)
        # model_trt = torch2trt(model,[random_input], ) # 设置网络的精度 fp32_mode=True

        logger.info("Start validation, Total sample: {:d}".format(len(self.val_loader)))
        # 如何计算分类别的IoU
        for i, (image, target, filename) in enumerate(self.val_loader):
            image = image.to(self.device)
            target = target.to(self.device)

            with torch.no_grad():
                outputs = model(image) 
                # outputs = model_trt(image) 
            self.metric.update(outputs[0], target)
            pixAcc, mIoU = self.metric.get()
            pred = torch.argmax(outputs[0], 1)

            # 为了计算每一类的mIoU
            hist = hist_info(target,pred,self.num_cls)
            self.hist += hist
            logger.info("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
                i + 1, pixAcc * 100, mIoU * 100))

            if self.args.save_pred:
                pred = torch.argmax(outputs[0], 1) # 保存预测的图像
                pred = pred.cpu().data.numpy()

                predict = pred.squeeze(0)
                # 保存预测图像
                # mask = get_color_pallete(predict, self.args.dataset)
                # mask.save(os.path.join(outdir, os.path.splitext(filename[0])[0] + '.png'))
                
                # 这里保存掩膜图像
                # img = ToPILImage()(image.squeeze(0)) # 图像经过变换，不对劲
                img = PIL.Image.open(self.root_path+'/JPEGImages/'+filename[0])
                scale_ratio = args.scale_ratio
                if scale_ratio != None:
                    w,h = img.size
                    img = img.resize((int(w*scale_ratio),int(h*scale_ratio)),PIL.Image.BILINEAR)
                img = np.array(img)
                out_viz_file = os.path.join(outdir, os.path.splitext(filename[0])[0] + '.jpg')
                viz = imgviz.label2rgb( label=predict,  # label=lbl
                                        img=imgviz.rgb2gray(img), #img
                                        font_size=10, label_names=self.label_name, loc="rb",)
                imgviz.io.imsave(out_viz_file, viz) # out_viz_file, viz
        scores, cls_iu  = compute_score(self.hist.cpu().numpy(),self.label_name)
        logger.info('{0}{1}'.format(scores,cls_iu))
        # logger.info("Average validation pixAcc: {:.3f}, all mIoU: {:.3f}".format(scores['pAcc'], scores['mIoU']))

        logger.info("Average validation pixAcc: {:.3f}, all mIoU: {:.4f}".format(pixAcc, mIoU))
        synchronize()


if __name__ == '__main__':
    args = parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
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

    # TODO: optim code
    args.save_pred = False
    if args.save_pred:
        outdir = '/data/zhangj/runs/pred_pic/{}_{}_{}'.format(args.model, args.backbone, args.dataset)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    logger = setup_logger("semantic_segmentation", args.log_dir, get_rank(),
                          filename='{}_{}_{}_log.txt'.format(args.model, args.backbone, args.dataset), mode='a+')

    evaluator = Evaluator(args)
    evaluator.eval()
    torch.cuda.empty_cache()
