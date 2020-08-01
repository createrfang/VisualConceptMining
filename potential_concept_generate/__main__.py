import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import PCGImageFolderWithCache
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as transform
from .salient_map import SalientMapGenerator
from .pcg import PotentialConceptGenerator
import argparse
import os
import pickle

trainset = '/share/home/fangzhengqing/Data/Keratitis_baseline/train'

Transform = transform.Compose([
    transform.Resize((224,224)),
    transform.ToTensor(),
])

def savepcg(cnt,path,target,sm,boxes,features):
    _ = os.makedirs(args.outputdir) if not os.path.exists(args.outputdir) else None
    filename = os.path.join(args.outputdir, f"{cnt}.pcg")
    with open(filename, "wb") as f:
        pickle.dump((path,target,sm,boxes,features),f)


def main(args):
    traindata = PCGImageFolderWithCache(trainset, transform=Transform)

    trainloader = DataLoader(traindata, 16)

    pcg = PotentialConceptGenerator()
    cnt = 0
    total = len(traindata.samples)
    for i, (data, odata, target, path) in enumerate(trainloader):
        data = data.cuda()
        sm, boxes, features = pcg(data, odata)
        for idx, smi in enumerate(sm):
            print(f"saving:{cnt}/{total}  {path[idx]}")
            savepcg(cnt,path[idx],target[idx],smi,boxes[idx],features[idx])
            cnt += 1

    return None

def parse_args():
    parser = argparse.ArgumentParser(description='potential concept generating')
    parser.add_argument('--traindata', metavar='DIR', help='path to trainset')
    parser.add_argument('--outputdir', metavar='DIR', help='path to pcg output')
    parser.add_argument('--pretraindir', metavar='DIR', help='path to output pretrainres')
    # parser.add_argument('logdir', metavar='DIR', help='path to pretrain log')
    parser.add_argument('--classes', default=4, type=int)
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    # writer = SummaryWriter(args.logdir)
    main(args)









