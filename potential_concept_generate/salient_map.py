from .Unet import UNet
from .GradCAM import GradCam
from .GuidedBP import GuidedBackpropReLUModel
import torch

from torchvision.models import DenseNet
import numpy as np

pwd = '/share/home/fangzhengqing/Code/VisualConceptMining'
unet_path = pwd+'/resource/Unet_cpu.model'
densenet_path = pwd+'/resource/keratitis_densenet201/pretrain_res/checkpoints/densenet121_33.pth'


# pretrainmodel = DenseNet(block_config=(6, 12, 48, 32),num_classes=4)
# pretrainmodel.load_state_dict(torch.load(densenet_path,map_location='cpu')["state_dict"])
# pretrainmodel.eval()
# pretrainmodel.cuda()

defaultconfig = {
    "pretrain_path": densenet_path,
    "use_cuda": True,
    "unet_path": unet_path
}


class  SalientMapGenerator:
    def __init__(self, smg_config=None):
        if not smg_config:
            smg_config = defaultconfig
        self.cammodel = DenseNet(block_config=(6, 12, 48, 32),num_classes=4)
        self.gbpmodel = DenseNet(block_config=(6, 12, 48, 32), num_classes=4)
        self.cammodel.load_state_dict(torch.load(smg_config["pretrain_path"], map_location='cpu')["state_dict"])
        self.gbpmodel.load_state_dict(torch.load(smg_config["pretrain_path"], map_location='cpu')["state_dict"])
        self.cammodel.eval()
        self.gbpmodel.eval()
        self.gradcam = GradCam(self.cammodel, "features.denseblock4", smg_config["use_cuda"])
        self.guidebp = GuidedBackpropReLUModel(self.gbpmodel, smg_config["use_cuda"])
        self.useunet = False
        if smg_config["unet_path"]:
            self.useunet = True
            self.unet = UNet(3, 3)
            self.unet.load_state_dict(torch.load(smg_config["unet_path"], map_location='cpu'))
            self.unet.eval()
            if smg_config["use_cuda"]:
                self.unet.cuda()

    def __call__(self,data):

        cam = self.gradcam(data)
        cam /= np.max(cam)
        gbp = self.guidebp(data)
        gbp = torch.sum(gbp, 0).detach().numpy()
        gbp /= np.max(gbp)
        sm = 3 * gbp + 2* cam
        if self.useunet:
            seg = self.unet(data).cpu()
            seg = torch.sum(seg, 1)[0].detach().numpy()
            seg /= np.max(seg)
            sm += seg
        sm/=np.max(sm)
        return sm



