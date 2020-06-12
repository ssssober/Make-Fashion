# -*- coding: utf-8 -*-
import os
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from datasets.data_io import get_transform, read_all_lines
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import cv2
# from skimage import io, color
from skimage.color import rgb2lab, rgb2gray, lab2rgb

################rgb2lab-L-train##############################3

class RGB2LDataLoad(Dataset):
    def __init__(self, datapath, list_filename, training, crop_h, crop_w, channels):
        self.datapath = datapath
        self.rgb_filenames = self.load_path(list_filename)
        self.training = training
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.channels = channels  #

    def load_path(self, list_filename):
        with open(list_filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        splits = [line.split() for line in lines]
        rgb_images = [x[0] for x in splits]
        return rgb_images

    # load:PILImage
    def load_PILimage(self, filename):
        if self.channels == 3:
            return Image.open(filename).convert('RGB')
        elif self.channels == 1:
            return Image.open(filename).convert('L')

    def __len__(self):
        return len(self.rgb_filenames)

    def __getitem__(self, index):
        # load all pil.png
        rgb_img = self.load_PILimage(os.path.join(self.datapath, self.rgb_filenames[index]))  # rgb.jpg

        # add left_png.name
        rgb_pathname = self.rgb_filenames[index]

        if self.training:
            # Training-setplace205-RGBData
            w, h = rgb_img.size
            #
            x1 = random.randint(0, w - self.crop_w)
            y1 = random.randint(0, h - self.crop_h)

            # randomly PIL.png crop: all png
            rgb_img = rgb_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            rgb_img = np.array(rgb_img)

            img_lab = rgb2lab(rgb_img)

            # L
            img_L = img_lab[:, :, 0]
            img_L = torch.from_numpy(img_L).float()
            img_L = img_L / 255.0
            img_L = torch.unsqueeze(img_L, dim=0)  # (1, h, w)

            # ab
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()  # (2, h, w)
            img_ab = (img_ab + 128) / 255.0  #

            # rgb--tensor
            img_rgb = torch.from_numpy(rgb_img.transpose((2, 0, 1))).float()

            return{"imgl": img_L,
                   "imgab": img_ab,
                   "imgrgb": img_rgb}
        else:
            w, h = rgb_img.size

            x1 = random.randint(0, w - self.crop_w)
            y1 = random.randint(0, h - self.crop_h)

            # randomly PIL.png crop: all png
            rgb_img = rgb_img.crop((x1, y1, x1 + self.crop_w, y1 + self.crop_h))
            rgb_img = np.array(rgb_img)  #

            if self.channels == 3:
                # rgb2lab:L
                img_lab = rgb2lab(rgb_img)
                # L
                img_L = img_lab[:, :, 0]  # (h, w)
                img_L = torch.from_numpy(img_L).float()
                img_L = img_L / 255.0
                img_L = torch.unsqueeze(img_L, dim=0)  # (1, h, w)

            else:
                #
                img_L = torch.from_numpy(rgb_img).float()
                img_L = img_L / 255.0
                img_L = torch.unsqueeze(img_L, dim=0)  # (1, h, w)

            return {"imgl": img_L,
                    "rgb_name": rgb_pathname}
