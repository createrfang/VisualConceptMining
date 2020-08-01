#pretrain a Densenet 121
import torch
import torch.nn.functional as F
from torchvision.models.densenet import DenseNet
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transform
import argparse
from torch.optim import Adam,SGD
from torch.optim.lr_scheduler import StepLR
from utils.util import AccMeter, AverageMeter
from utils.dataloader import ImageFolderWithCache
import os
import time


Transform = transform.Compose([
    transform.Resize((224,224)),
    transform.ToTensor(),
])


def main(args):
    _ = [os.mkdir(args.pretraindir) if not os.path.exists(args.pretraindir) else None]
    if not os.path.exists(os.path.join(args.pretraindir,'checkpoints')):
        os.mkdir(os.path.join(args.pretraindir,'checkpoints'))
    traindata = ImageFolderWithCache(args.traindata, transform=Transform)
    trainloader = DataLoader(traindata, 32, shuffle=True)
    testdata = ImageFolderWithCache(args.testdata, transform=Transform)
    testloader = DataLoader(testdata, 32, shuffle=True)
    model = DenseNet(block_config=(6, 12, 48, 32), num_classes=args.classes)
    model.cuda()
    optim = SGD(
        model.parameters(),
        lr=1e-1, momentum=0.4, weight_decay=1e-3
    )

    lch = StepLR(optim, 30, 0.1, -1)
    accmeter = AccMeter()
    for i in range(args.epoch):
        mat, loss = train(trainloader, model, i, optim)
        lch.step()
        accmeter.compute(mat)
        print(f"Epoch{i} finished, avgloss:{loss:.6f}, totalacc:{accmeter.totalacc:.6f}, nacc:{accmeter.nacc}")
        print(mat)
        writer.add_scalar('Train/acc',accmeter.totalacc,i)
        writer.add_scalar('Train/losses',loss,i)
        tmat, tloss = test(testloader, model)
        accmeter.compute(tmat)
        print(f"Test finished, testloss:{tloss:.6f}, totalacc:{accmeter.totalacc:.6f}, nacc:{accmeter.nacc}")
        print(tmat)
        writer.add_scalar('Test/acc',accmeter.totalacc,i)
    pass


def compute_acc(out,target):
    n,d = out.shape
    res = torch.zeros((d,d))
    pred = torch.argmax(out,1)
    right = 0
    for i in range(n):
        res[pred[i]][target[i]]+=1

    for i in range(d):
        right+=res[i][i]

    return res, right/torch.sum(res)


def train(traindataloader, model, epoch, optim):
    Ressultmat = None
    # switch to train mode
    model.train()
    losses = AverageMeter()
    path = os.path.join(
        args.pretraindir,
        'checkpoints',
        'densenet121_' + str(epoch) + '.pth',
    )

    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict()
    }, path)

    end = time.time()

    for i,d in enumerate(traindataloader):
        losses = AverageMeter()
        data,target = d
        out = model(data.cuda())
        out = F.log_softmax(out,1)
        loss = F.nll_loss(out,target.cuda())

        # record loss
        losses.update(loss.item(), out.size(0))

        optim.zero_grad()
        loss.backward()
        optim.step()

        resmat,acc = compute_acc(out, target)

        if Ressultmat is None:
            Ressultmat=resmat
        else:
            Ressultmat+=resmat

        print(f"loss:{loss.cpu().data.item()}, acc:{acc}, time:{time.time() - end}")
        # measure elapsed time
        end = time.time()

        writer.add_scalar('Trainepoch/loss', losses.avg,epoch*len(traindataloader)+i)
    return Ressultmat,losses.avg

def test(testdataloader, model):

    losses = AverageMeter()

    Ressultmat = None
    # switch to eval mode
    model.eval()

    end = time.time()

    with torch.no_grad():
        for i,d in enumerate(testdataloader):
            data,target = d
            out = model(data.cuda())
            out = F.log_softmax(out,1)
            loss = F.nll_loss(out,target.cuda())
            # record loss
            losses.update(loss.item(), out.size(0))
            resmat,acc = compute_acc(out,target)
            if Ressultmat is None:
                Ressultmat=resmat
            else:
                Ressultmat+=resmat

            print(f"loss:{loss.cpu().data.item()}, acc:{acc}, time:{time.time() - end}")
            # measure elapsed time
            end = time.time()

        return Ressultmat,losses.avg

def parse_args():
    parser = argparse.ArgumentParser(description='Pretrain arguments for densenet')
    parser.add_argument('traindata', metavar='DIR', help='path to trainset')
    parser.add_argument('testdata', metavar='DIR', help='path to testset')
    parser.add_argument('pretraindir', metavar='DIR', help='path to output pretrainres')
    parser.add_argument('logdir', metavar='DIR', help='path to pretrain log')
    parser.add_argument('--classes', default=4, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    writer = SummaryWriter(args.logdir)
    main(args)
