import numpy as np
import torch
import cv2
from .salient_map import SalientMapGenerator
from .anchor_generator import AnchorGenerator
#from .clustering import Kmeans
from skimage import feature
import numpy as np
from faiss import Kmeans, IndexFlatL2

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        # return the histogram of Local Binary Patterns
        return lbp

    def gethist(self,lbp, eps=1e-7):
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        return hist

anchor_sizes = (32, 64, 128)
aspect_ratios = (0.5, 1.0, 2.0)
anchor_stride = (8, 16, 32)
straddle_thresh = 0

no_unet_config = {
    "pretrain_path": '/share/home/fangzhengqing/Code/VisualConceptMining/resource/keratitis_densenet201/pretrain_res/checkpoints/densenet121_33.pth',
    "use_cuda": True,
    "unet_path": None
}

class PotentialConceptGenerator(torch.nn.Module):
    rLBP=LocalBinaryPatterns(24,8)
    gLBP = LocalBinaryPatterns(24, 8)
    bLBP = LocalBinaryPatterns(24, 8)
    def __init__(self):
        super(PotentialConceptGenerator, self).__init__()
        self.smg = SalientMapGenerator(no_unet_config)
        self.acg = AnchorGenerator(anchor_sizes, aspect_ratios, anchor_stride, straddle_thresh)

    def forward(self, data: torch.Tensor, origindata: torch.Tensor):
        assert data.shape[0] == origindata.shape[0]
        self.rate = int(origindata.shape[2]/data.shape[2])
        smap = []
        for i in range(0, data.shape[0]):
            datai = torch.unsqueeze(data[i], 0)
            smapi = self.smg(datai)
            smap.append(smapi)
        smap = torch.Tensor(smap)
        anchors = self.acg(data, smap)
        salient_anchors = self.salient_anchor_filtering(smap, anchors)
        print(len(salient_anchors))
        salient_anchors, features = self.most_representive_patches(salient_anchors,origindata)
        return smap, salient_anchors, features

    def salient_anchor_filtering(self, smap, anchors):
        salient_anchors = []
        for i, sm in enumerate(smap):
            acs = anchors[i]
            allacs = []
            for ac in acs:
                for box in ac.bbox:
                    box = box.int()
                    patch = sm[box[0]:box[2], box[1]:box[3]]
                    allacs.append((box, torch.var(patch), torch.mean(patch)))
            n = len(allacs)
            allacs.sort(key=lambda x: x[1], reverse=True)
            varlist = list(zip(*allacs))[0][0:n // 2]
            allacs.sort(key=lambda x: x[2], reverse= True)
            meanlist = list(zip(*allacs))[0][0:n // 2]
            sanchors = list(set(varlist).intersection(set(meanlist)))
            salient_anchors.append(sanchors)
        return salient_anchors


    def most_representive_patches(self, anchors, odata):
        res_anchor=[]
        features = []
        for i, o in enumerate(odata):
            rlbp = self.rLBP.describe(o[0])
            glbp = self.gLBP.describe(o[1])
            blbp = self.bLBP.describe(o[2])
            acs = anchors[i]
            acs = [self.rate*ac for ac in acs]
            f = []
            for ac in acs:
                r = rlbp[ac[0]:ac[2], ac[1]:ac[3]]
                vr = self.rLBP.gethist(r)
                g = glbp[ac[0]:ac[2], ac[1]:ac[3]]
                vg = self.gLBP.gethist(g)
                b = blbp[ac[0]:ac[2], ac[1]:ac[3]]
                vb = self.bLBP.gethist(b)
                v = np.concatenate([vr,vg,vb],0)
                f.append(v)
            f = np.array(f).astype('float32')

            kmeans = Kmeans(f.shape[1],10)
            Kmeans.gpu = True
            kmeans.train(f)
            index = IndexFlatL2(f.shape[1])
            index.add(f)
            D,I = index.search(kmeans.centroids, 1)
            I = np.squeeze(I,1)
            print(I)
            fis = [f[index] for index in I]
            boxes = [acs[index] for index in I]
            res_anchor.append(boxes)
            features.append(fis)
        return res_anchor, features
