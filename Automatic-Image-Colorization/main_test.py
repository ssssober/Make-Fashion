# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
from PIL import Image
from skimage import io, color
import matplotlib.pyplot as plt

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='colornet')
parser.add_argument('--mode', type=str, default='test', help='train or test')
parser.add_argument('--model', default='colornet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default='', help='data path')
parser.add_argument('--channels', type=int, default=3, help='rgb input channels')
parser.add_argument('--in_channels', type=int, default=1, help='net input channels')
parser.add_argument('--out_channels', type=int, default=2, help='net output channels')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--test_crop_height', type=int, default=128, help='test crop height')
parser.add_argument('--test_crop_width', type=int, default=256, help='test crop width')
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--saveresult', default='', help='save result images')

# parse arguments, set seeds
args = parser.parse_args()
os.makedirs(args.saveresult, exist_ok=True)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False, args.test_crop_height, args.test_crop_width, args.channels)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=1, drop_last=False)

# model, optimizer
model = __models__[args.model](args.in_channels, args.out_channels)
model = nn.DataParallel(model)
model.cuda()

# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()
    imgl = sample['imgl']
    imgl = imgl.cuda()
    # add name
    rgb_name = sample['rgb_name']
    torch.cuda.synchronize()
    start_time = time.time()
    pre = model(imgl)
    torch.cuda.synchronize()
    end_time = time.time()
    time_consume = end_time - start_time
    print('Time_Consume: ', time_consume * 1000)
    image_outputs = {"imgl": imgl, "pre": pre, "rgb_name": rgb_name}
    return image_outputs

def test_batch():
    # find all checkpoints file and sort according to epoch id
    #all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".tar")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    for ckpt_idx, ckpt_path in enumerate(all_saved_ckpts):
        loadckpt = os.path.join(args.logdir, ckpt_path)
        print("loading the model in logdir: {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        #model.load_state_dict(state_dict['model'])
        model.load_state_dict(state_dict['state_dict'])

        for batch_idx, sample in enumerate(TestImgLoader):
            image_outputs = test_sample(sample, compute_metrics=False)

            # add left_png.name
            rgb_name = image_outputs["rgb_name"][0]
            save_name = rgb_name.split("/")[-1]
            save_name = save_name[:-4]
#############################################################################################################
            # save pre_rgb.jpg
            img_L = image_outputs["imgl"][0]  # (1, h, w)
            img_L = img_L * 255  # rgb2lab--L
            # img_L = img_L * 100  # rgb2gray--gray: gray2L(*100)
            img_L = torch.squeeze(img_L, 0)
            np_imgL = img_L.detach().cpu().numpy()

            # ab results
            out_ab = image_outputs["pre"][0]
            out_ab = out_ab * 255 - 128
            out_ab = out_ab.detach().cpu().numpy()
            out_ab = out_ab.transpose(1, 2, 0)

            # concat:L and ab -- to -- rgb
            img_lab_out = np.concatenate((np_imgL[:, :, np.newaxis], out_ab), axis=2)
            img_rgb_out = color.lab2rgb(img_lab_out.astype(np.float64))
            plt.imsave(args.saveresult + '{}_rgb.jpg'.format(save_name), img_rgb_out)
##################################################################################################

            # del img_L
            del out_ab
            del img_rgb_out
        gc.collect()

if __name__ == '__main__':
    if args.mode == 'test':
        test_batch()
