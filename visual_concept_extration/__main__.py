import os

import argparse
import os
import pickle
import time

import faiss
import numpy as np

import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.dataloader import PCGDataset

PCPATH="/share/home/fangzhengqing/Data/Keratitisbaseline_pcg_no_unet2/train"

if __name__=='__main__':
    traindata = PCGDataset(PCPATH)
    print(traindata[0])