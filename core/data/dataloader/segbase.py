"""Base segmentation dataset"""
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

__all__ = ['SegmentationDataset']


class SegmentationDataset(object):
    """Segmentation Base Dataset"""

    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480,pad_size=None,scale_ratio=0.75):
        super(SegmentationDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.pad_size = pad_size
        self.scale_ratio = scale_ratio

    def _val_sync_transform(self, img, mask): # 验证集不处理
        # outsize = self.crop_size
        # short_size = outsize
        # w, h = img.size
        # if w > h:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # else:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # # center crop
        # w, h = img.size
        # x1 = int(round((w - outsize) / 2.))
        # y1 = int(round((h - outsize) / 2.))
        # img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        # mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # 将图片按照缩放比例缩放
        scale_ratio = self.scale_ratio
        if scale_ratio != None:
            w,h = img.size
            img = img.resize((int(w*scale_ratio),int(h*scale_ratio)),Image.BILINEAR)
            mask = mask.resize((int(w*scale_ratio),int(h*scale_ratio)),Image.BILINEAR)
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # 如果base_size和crop_size是None就不resize和裁剪
        # 先将图片padding到一个固定尺寸的正方形
        if self.pad_size:
        #     w,h = img.size
        #     pad_w = max(self.pad_size - w, 0)
        #     pad_h = max(self.pad_size - h, 0)
        #     img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0) # border=(left, top, right, bottom)
        #     mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)
            # 缩放图像到统一大小
            img = img.resize((self.pad_size,self.pad_size),Image.BILINEAR)
            mask = mask.resize((self.pad_size,self.pad_size),Image.BILINEAR)
        # random mirror # 随机镜像翻转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge) # 随机缩放0.5-2.0
        if self.base_size:
            # 这里缩放时不用base_size,根据原始图片缩放
            # short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            w, h = img.size # 所以basesize是以短的边为标准吗
            short = min(w,h)
            short_size = random.randint(int(short * 0.5), int(short * 2.0))
            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)
        # 将图片按照缩放比例缩放
        scale_ratio = self.scale_ratio
        if scale_ratio != None:
            w,h = img.size
            img = img.resize((int(w*scale_ratio),int(h*scale_ratio)),Image.BILINEAR)
            mask = mask.resize((int(w*scale_ratio),int(h*scale_ratio)),Image.BILINEAR)

        # pad crop #裁剪的尺寸是参数指定的
        if self.crop_size:
            if short_size < crop_size:
                padh = crop_size - oh if oh < crop_size else 0
                padw = crop_size - ow if ow < crop_size else 0
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
                mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
            # random crop crop_size
            w, h = img.size
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)
            img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
            mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP 这个有什么用？先注释看看
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        return np.array(mask).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0
