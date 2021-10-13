"""Pascal VOC Semantic Segmentation Dataset."""
import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset


class MTVOCSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'MT-defect-voc'
    NUM_CLASS = 6

    def __init__(self, root='/data/zhangj/voc', split='train', mode=None, transform=None, **kwargs):
        super(MTVOCSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        _auxiliary_dir = os.path.join(_voc_root, 'AuxiliaryGT') # 加的
        self.voc_root = _voc_root
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        elif split == 'trainval':
            _split_f = os.path.join(_splits_dir, 'trainval.txt') # 加的,8:2训练
        else:
            raise RuntimeError('Unknown dataset split.')

                # 读取semantic_labels.txt文件
        se_labels_dt = {}
        _sells_f = os.path.join(_auxiliary_dir, '1semantic_labels.txt')
        with open(_sells_f,'r') as lines:
            for line in lines:
                line = line.rstrip('\n')
                linel = line.split('\t')
                fn,lb = linel[0],eval(linel[1])
                se_labels_dt[fn] = torch.tensor(lb)


        self.images = []
        self.masks = []
        self.masks_edge = []
        self.se_labels = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                # if split != 'test':
                _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".png")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
                _mask_edge = os.path.join(_auxiliary_dir, line.rstrip('\n') + ".png") # 加的
                self.masks_edge.append(_mask_edge)
                self.se_labels.append(se_labels_dt[line.rstrip('\n')])

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), _voc_root))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index]) # 打开标注图像？
        mask_edge = Image.open(self.masks_edge[index])
        # synchronized transform 这步是做什么？
        if self.mode == 'train':
            img0 = img
            img, mask = self._sync_transform(img, mask)
            _, mask_edge = self._sync_transform(img0, mask_edge)
            se_label = self.se_labels[index]
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
            if self.transform is not None:
                img = self.transform(img)
            return img, mask,os.path.basename(self.images[index])
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask,mask_edge,se_label, os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        return ('background', 'blowhole', 'break', 'crack', 'fray', 'uneven')


if __name__ == '__main__':
    dataset = MTVOCSegmentation()