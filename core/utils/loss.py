"""Custom losses."""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from torch.autograd import Variable
from .logger import setup_logger
from .distributed import get_rank
from .DICELoss import BinaryDiceLoss

__all__ = ['MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'EncNetLoss', 'ICNetLoss', 'get_segmentation_loss']


# TODO: optim function
# Hi，the default loss function is cross entropy loss，
# MixSoftmaxCrossEntropyLoss means auxiliary loss （deep supervised）+ cross entropy loss.
class MixSoftmaxCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, aux=True, aux_weight=0.2, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, *inputs, **kwargs):
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs, **kwargs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyLoss, self).forward(*inputs))


# reference: https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/encoding/nn/loss.py
class EncNetLoss(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with SE Loss"""

    def __init__(self, se_loss=True, se_weight=0.2, nclass=19, aux=False,
                 aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(EncNetLoss, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        if not self.se_loss and not self.aux:
            return super(EncNetLoss, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            return dict(loss=loss1 + self.aux_weight * loss2)
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(EncNetLoss, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.se_weight * loss2)
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(EncNetLoss, self).forward(pred1, target)
            loss2 = super(EncNetLoss, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return dict(loss=loss1 + self.aux_weight * loss2 + self.se_weight * loss3)

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass - 1)
            vect = hist > 0
            tvect[i] = vect
        return tvect




# 这个ohem定义是为cityscapes写的，修改以适应自己的数据集
class OhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_index=-1, thresh=0.7, min_kept=100000, use_weight=False, **kwargs):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            # weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754,
            #                             1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
            #                             1.0865, 1.1529, 1.0507]) # cityscapes有19类，这个系数是怎么得到的
            # 这里权重怎么设很重要，否则无法收敛
            # weight = torch.FloatTensor([0.9655,0.0001503,0.002974,0.0003488,0.006795,0.02421]) # 这里修改成了mt-voc每类像素点的频率
            # weight = torch.FloatTensor([1.0357, 6653.36, 336.2475, 2908.67, 2866.97, 41.3052])  # 这里权重应该为比例的倒数！！！
            weight = torch.FloatTensor([1,  2614, 132, 1126, 57,16])
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, pred, target):
        n, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if self.min_kept > num_valid:
            print("Lables: {}".format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~ valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)

        return self.criterion(pred, target)


class MixSoftmaxCrossEntropyOHEMLoss(OhemCrossEntropy2d):
    def __init__(self, aux=False, aux_weight=0.4, weight=None, ignore_index=-1, **kwargs):
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(ignore_index=ignore_index)
        self.aux = aux
        self.aux_weight = aux_weight
        # self.bceloss = nn.BCELoss(weight)

    def _aux_forward(self, *inputs, **kwargs): # 这个辅助loss是干什么的
        *preds, target = tuple(inputs)

        loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[0], target)
        for i in range(1, len(preds)):
            aux_loss = super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(preds[i], target)
            loss += self.aux_weight * aux_loss
        return loss

    def forward(self, *inputs,**kwargs): #self, *inputs
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])
        # 不懂上面这两行的作用
        if self.aux:
            return dict(loss=self._aux_forward(*inputs))
        else:
            return dict(loss=super(MixSoftmaxCrossEntropyOHEMLoss, self).forward(*inputs))

# TODO: optim function
class ICNetLoss(nn.CrossEntropyLoss): # nn.CrossEntropyLoss
    """Cross Entropy Loss for ICNet"""

    def __init__(self, nclass=4, aux_weight=0.4, ignore_index=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.nclass = nclass
        self.aux_weight = aux_weight
        self.cri = OhemCrossEntropy2d()

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # print(pred.shape, pred_sub4.shape, pred_sub8.shape, pred_sub16.shape, target.shape)
        # [batch, W, H] -> [batch, 1, W, H]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        # loss1 = self.cri(pred_sub4, target_sub4)
        # loss2 = self.cri(pred_sub8, target_sub8)
        # loss3 = self.cri(pred_sub16, target_sub16)
        return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)

# 默认使用OHEM
class FDSNetLoss(OhemCrossEntropy2d):
    """Cross Entropy Loss for ICNet"""

    def __init__(self, aux=True,aux_weight=0.4, ignore_index=-1, **kwargs):
        super(FDSNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight
        self.aux = aux
        self.bceloss = nn.BCELoss()
        self.diceloss = BinaryDiceLoss()
        # self.logger = setup_logger("auxiliary",'/data/zhangj/runs/logs/', get_rank(), filename='{}_log.txt'.format(
        # 'auxiliary_loss_citys'))


    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + list(target))

        pred, saliency,sevector, target,target_edge,selabels = tuple(inputs)
        # [batch, W, H] -> [batch, 1, W, H]

        loss1 = super(FDSNetLoss, self).forward(pred, target)
        # 将语义分割的标注图转边缘信息的黑白图监督
        # 1.这里彩色标注图转黑白图
        # target_sa = (target != 0).float()
        # # 2.标注图转边缘图
        # target_edge = self.getLabel2Edge(target.cpu().numpy())
        # target_edge = torch.from_numpy(target_edge)
        # target_edge = (target_edge!=0).float()#.cuda()

        # 3.油(标签是1)转边缘图，划痕3和斑点2转黑白图
        # 反过来，油用显著图，划痕和斑点用边缘图
        # target_oil = (target == 1).to(torch.int64)
        # target_other = target - target_oil
        # target_oil_edge = self.getLabel2Edge(target_oil.cpu().numpy())
        # target_oil_edge = torch.from_numpy(target_oil_edge)
        # target_oil_edge = target_oil_edge.int().cuda()
        # target_ss = (target_other != 0).int()
        # target_all = target_oil_edge + target_ss
        # target_all = (target_all != 0).float()

        # 得到边缘监督图
        try:
            saliency = torch.squeeze(saliency)
            target_edge = (target_edge > 0).float().cuda()
            loss2 = self.bceloss(saliency, target_edge)
            # loss2 = self.diceloss(saliency, target_edge) # 测试使用DICE loss的效果

            # 标注图得到一个类别向量表示是否出现
            sevector =  F.softmax(sevector,dim=1)
            selabels = selabels.float().cuda()
            loss3 = self.bceloss(sevector,selabels)
        except:
            print('!!!!error happend in loss')

        return  {'loss1':loss1,'loss2':loss2,'loss3':loss3}
        # return dict(loss=loss1 + loss2 * self.aux_weight+loss3*0.5  ) 


# 增加的focal loss
class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,alpha=0.25,reduction='mean',**kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # self.alpha = alpha
        # self.alpha = Variable(torch.ones(5, 1))
        # self.alpha = Variable(torch.tensor([0.1,0.25,0.25,0.15,0.25]))
        # self.alpha = Variable(torch.tensor([1.0357, 6653.36, 336.2475, 2908.67, 2866.97, 41.3052])) # 这里设置的参数是mt-voc
        # self.alpha = Variable(torch.tensor([0.01, 0.4, 0.1, 0.4, 0.1, 0.1])) # 这里设置的参数是mt-voc
        self.alpha = Variable(torch.tensor([1.1, 51, 13, 42])) # 这里设置的参数是sd-voc
        self.alpha = self.alpha.unsqueeze(dim=1)
        self.reduction = reduction

    def forward(self,pred,targets):
        # pred:    torch.Size([2, 5, 1080, 1920])
        # targets: torch.Size([2, 1080, 1920])
        inputs = pred[0] # inputs=pred
        # print('------------inputs',inputs)
        targets = targets.unsqueeze(dim=1)
        # print('------------targets',targets)
        # print('-------------Focal loss,inputs',type(inputs),len(inputs),type(inputs[0]),len(inputs[0]))
        # print(type(targets),len(targets))

        # N = inputs.size(0)
        # C = inputs.size(1)
        # P = F.softmax(inputs)

        # target : N, 1, H, W
        inputs = inputs.permute(0, 2, 3, 1)
        targets = targets.permute(0, 2, 3, 1)
        num, h, w, C = inputs.size()
        N = num * h * w
        inputs = inputs.reshape(N, -1)   # N, C
        targets = targets.reshape(N, -1)  # 待转换为one hot label
        P = F.softmax(inputs, dim=1)  # 先求p_t
        # print('------------P',P)


        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)  # 得到label的one_hot编码

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()  # 如果是多GPU训练 这里的cuda要指定搬运到指定GPU上 分布式多进程训练除外
        alpha = self.alpha[ids.data.view(-1)]
        # y*p_t  如果这里不用*， 还可以用gather提取出正确分到的类别概率。
        # 之所以能用sum，是因为class_mask已经把预测错误的概率清零了。
        probs = (P * class_mask).sum(1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)
        # print('------------probs',probs)
        # y*log(p_t)
        log_p = probs.log()
        # -a * (1-p_t)^2 * log(p_t)
        # print('------------log_p',log_p)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('------------batch_loss',batch_loss)
        if self.reduction=='mean':
            loss0 = batch_loss.mean()
        elif self.reduction=='sum':
            loss0 = batch_loss.sum()
        return dict(loss=loss0)


class Multi_CELoss(nn.Module):
    def __init__(self,use_ohem=False, use_focal=False, **kwargs):
        super(Multi_CELoss,self).__init__()
        if use_ohem:
            self.cri = MixSoftmaxCrossEntropyOHEMLoss()
        else:
            self.cri = nn.CrossEntropyLoss()


    # u2netp的loss
    def muti_bce_loss_fusion(self, d0, d1, d2, d3, d4, d5, d6, labels_v):
        # bce_loss = nn.BCELoss(size_average=True)
        bce_loss = self.cri
        loss0 = bce_loss(d0,labels_v)
        loss1 = bce_loss(d1,labels_v)
        loss2 = bce_loss(d2,labels_v)
        loss3 = bce_loss(d3,labels_v)
        loss4 = bce_loss(d4,labels_v)
        loss5 = bce_loss(d5,labels_v)
        loss6 = bce_loss(d6,labels_v)

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return loss0, loss

    def forward(self,pred,targets):
        d0, d1, d2, d3, d4, d5, d6 = pred
        loss2, loss0 = self.muti_bce_loss_fusion(d0, d1,d2,d3,d4,d5,d6,targets)

        return dict(loss=loss0)



def get_segmentation_loss(model, use_ohem=False,use_focal=False, **kwargs):
    model = model.lower()
    if model == 'encnet':
        return EncNetLoss(**kwargs)
    elif model == 'icnet':
        return ICNetLoss(**kwargs)
    elif model == 'u2netp':
        return Multi_CELoss(False,use_focal,**kwargs)
    elif ( model == 'fdsnet')and kwargs['aux']:
        return FDSNetLoss(**kwargs)

    if use_ohem:
        return MixSoftmaxCrossEntropyOHEMLoss(**kwargs)
    if use_focal:
        return FocalLoss(**kwargs)
    else:
        return MixSoftmaxCrossEntropyLoss(**kwargs)
