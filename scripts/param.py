# Test model parameters and FPS
import argparse
import time
import datetime
import os
import shutil
import sys
import torch
from torch2trt import torch2trt

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

from core.models.fdsnet0 import get_fdsnet0
from core.models.fdsnet import get_fdsnet
from core.models.u2net import get_u2netp
from core.models.espnet import get_espnet
from core.models.fast_scnn import get_fast_scnn
from core.models.dfanet import get_dfanet
from core.models.enet import get_enet
from  core.models.icnet import get_icnet
from core.models.bisenet import get_bisenet
from core.models.lednet import get_lednet

from core.models.fcn import get_fcn8s
from core.models.pspnet import get_psp
from core.models.psanet import get_psanet
from core.models.deeplabv3plus import get_deeplabv3plus
from core.models.emanet import get_emanet


cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)


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
    args = parser.parse_args()
    return args

name_model = {
                'fdsnet':get_fdsnet,
                'fastscnn':get_fast_scnn,
                'fcn':get_fcn8s,
                'enet':get_enet,
                'icnet':get_icnet,
                'bisenet':get_bisenet,
                'bisenetv2':'',
                'espnet':get_espnet,
                'dfanet': get_dfanet,
                'u2netp':get_u2netp,
                'deeplabv3plus':get_deeplabv3plus,
                'fcn8s':get_fcn8s,
                'psp':get_psp,
                'psanet':get_psanet,
                'lednet':get_lednet,
                'emanet':get_emanet,
                }


def param():
    args = parse_args()
    
    # from thop import profile
    # from thop import clever_format

    # img = torch.randn(1, 3, 320, 320)
    get_model = name_model[args.model]
    model = get_model('phone_voc')
    # model.eval()
    # macs, params = profile(model, inputs=(img,))
    # print('Total macc:{}, Total params: {}'.format(macs, params))
    # # # macs, params = clever_format([macs, params], "%.3f")

    from torchinfo import summary
    # # from torchscan import summary

    # summary(model, ( 1,3, 1440, 810))

    # test FPS
    from tqdm import tqdm
    import time
    # cuDnn configurations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    
    device_id = 0
    random_input = torch.randn(1, 3, 319, 319).to('cuda:{}'.format(device_id))
    # random_input = torch.randn(1, 3, 450, 450).to('cuda:{}'.format(device_id))
    # random_input = torch.randn(1, 3, 201, 201).to('cuda:{}'.format(device_id))
    # random_input = torch.randn(1, 3, 1440, 810).to('cuda:{}'.format(device_id))
    
    model = model.to('cuda:{}'.format(device_id))
    model.eval()

    # tensorRT
    # model_trt = torch2trt(model,[random_input],fp32_mode=True)
    # print('Already get tensorRT model!!')

    time_list = []
    for i in tqdm(range(10001)):
        torch.cuda.synchronize()
        tic = time.time()
        model(random_input)
        # model_trt(random_input)
        torch.cuda.synchronize()
        # the first iteration time cost much higher, so exclude the first iteration
        #print(time.time()-tic)
        time_list.append(time.time()-tic)
    time_list = time_list[1:]
    print("     + Done 10000 iterations inference !")
    print("     + Total time cost: {}s".format(sum(time_list)))
    print("     + Average time cost: {}s".format(sum(time_list)/10000))
    print("     + Frame Per Second: {:.2f}".format(1/(sum(time_list)/10000)))

param()