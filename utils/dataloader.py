from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

import pickle
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

from torchvision.datasets import ImageFolder
from PIL import Image
import os

PVCTransform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
)

#potential visual concept loader
def pvc_loader(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        idx, data, target, path, salientpatches, predres = d
        splist = []
        boxes = []
        for sp in salientpatches:
            img = Image.fromarray(sp[1])
            spdata = PVCTransform(img).numpy()
            splist.append(spdata)
            boxes.append(sp[0].cpu().numpy())
        sps = torch.from_numpy(np.array(splist))
        boxes = np.array(boxes)
        data = torch.squeeze(data,0)
        return data,sps,target,path,boxes


class VCMDataloader(ImageFolder):
    def __init__(self,root):
        super(VCMDataloader, self).__init__(root, loader=pvc_loader)
        cache_root = root+'_cache'


class ImageFolderWithCache(ImageFolder):
    def __init__(self, dataroot=None, transform=None):
        self.trans = transform
        super(ImageFolderWithCache, self).__init__(dataroot, transform=self.trans)
        self.cacheroot = dataroot+'_cache'

    def __getitem__(self, index):
        if self.cacheroot:
            _ = os.mkdir(self.cacheroot) if not os.path.exists(self.cacheroot) else None
            sep = os.path.sep
            path, _ = self.samples[index]
            category, name = path.split(sep)[-2], path.split(sep)[-1]
            name = name.split('.')[0]
            cacheurl = self.cacheroot + sep + category + sep + name + '.pkl'
            if os.path.exists(cacheurl):
                sample = pickle.load(open(cacheurl, 'rb'))
            else:
                sample = super(ImageFolderWithCache, self).__getitem__(index)
                categorydir = self.cacheroot + sep + category
                _ = os.mkdir(categorydir) if not os.path.exists(categorydir) else None
                pickle.dump(sample, open(cacheurl, 'wb'))
        else:
            sample = super(ImageFolderWithCache, self).__getitem__(index)

        return sample

class PCGImageFolderWithCache(ImageFolder):
    totensor = transforms.Compose([
        transforms.Resize((1784,1784)),
        transforms.ToTensor()
    ])
    def __init__(self, dataroot=None, transform=None):
        self.transform = transform
        super(PCGImageFolderWithCache, self).__init__(dataroot, transform=self.transform)
        self.cacheroot = dataroot+'_pcgcache'


    def __getitem__(self, index):
        if self.cacheroot:
            _ = os.mkdir(self.cacheroot) if not os.path.exists(self.cacheroot) else None
            sep = os.path.sep
            path, _ = self.samples[index]
            category, name = path.split(sep)[-2], path.split(sep)[-1]
            name = name.split('.')[0]
            cacheurl = self.cacheroot + sep + category + sep + name + '.pkl'
            if os.path.exists(cacheurl):
                sample = pickle.load(open(cacheurl, 'rb'))
            else:
                sample = self.getitem(index)
                categorydir = self.cacheroot + sep + category
                _ = os.mkdir(categorydir) if not os.path.exists(categorydir) else None
                pickle.dump(sample, open(cacheurl, 'wb'))
        else:
            sample = self.getitem(index)

        return sample

    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def getitem(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.pil_loader(path)
        sample1 = sample
        sample2 = self.totensor(sample)
        if self.transform is not None:
            sample1 = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample1, sample2, target, path


class PCGDataset(Dataset):
    def __init__(self, path):
        super(PCGDataset, self).__init__()
        self.root = path
        self.samples = [os.path.join(path, p) for p in os.listdir(path)]
        self.samples.sort(key=lambda x: x.split(os.sep)[-1])

    def __getitem__(self, index):
        p = self.samples[index]
        with open(p, "rb") as f:
            (path, target, sm, boxes, features) = pickle.load(f)
        return (path, target, sm, boxes, features)

    def __len__(self):
        return len(self.samples)






