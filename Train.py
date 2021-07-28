import numpy as np
import torch
import torch.nn as nn
from Train_Feeder import Train_Feeder
from torch.utils.data import DataLoader
from GCN import gcn
from sklearn.metrics import precision_score, recall_score
from Meters import AverageMeter
from tensorboardX import SummaryWriter
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def make_labels(gtmat):
    return gtmat.view(-1)

def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    # print("GroundTruth:\t",label,"Pred:\t",pred)
    r = recall_score(label, pred)
    return p,r,acc

if __name__ == '__main__':

    writer=SummaryWriter('/Users/duan/Desktop/log')
    losses = AverageMeter()
    accs  = AverageMeter()
    precisions  = AverageMeter()
    recalls  = AverageMeter()


    val_losses=AverageMeter()
    val_accs =AverageMeter()

    trainset = Train_Feeder("/Users/duan/Desktop/ClustRead/data/S7/features.npy",
                      "/Users/duan/Desktop/ClustRead/data/S7/KNN.npy",
                      "/Users/duan/Desktop/ClustRead/data/S7/train_index.npy",
                      "/Users/duan/Desktop/ClustRead/data/S7/train_KNN.npy",
                      "/Users/duan/Desktop/ClustRead/data/S7/train_label.npy",
                      )
    trainloader = DataLoader(trainset, batch_size=4,shuffle=True)

    #构建网络，确定优化方法和损失函数
    net = gcn()
    opt = torch.optim.SGD(net.parameters(), 1e-2, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    net.train()

    #开始训练
    for epoch in range(4):
        for i, ((feat, adj,cid, h1id), gtmat) in enumerate(trainloader):

            pred = net(feat, adj,h1id)
            labels = make_labels(gtmat).long()
            loss = criterion(pred, labels)


            p, r, acc = accuracy(pred, labels)  #计算精确率、召回率、准确率

            opt.zero_grad()  #把由上一个batch计算得到的梯度清空
            loss.backward()  #计算梯度
            opt.step()       #更新模型

            losses.update(loss.item(),feat.size(0))
            accs.update(acc.item(),feat.size(0))
            precisions.update(p, feat.size(0))
            recalls.update(r,feat.size(0))

            writer.add_scalar("Train Loss", loss.data.item(), epoch * len(trainloader) + i)
            writer.add_scalar("Train Accuracy", acc.data.item(), epoch * len(trainloader) + i)
            writer.add_scalar("Train Precision", p, epoch * len(trainloader) + i)
            writer.add_scalar("Train Recalls", r, epoch * len(trainloader) + i)
            if i%20==0:
                print('Loss in Training Set:\t{losses.val:.3f} ({losses.avg:.3f})\t'
                      'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                      'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                      'Recall {recalls.val:.3f} ({recalls.avg:.3f})'
                    .format(losses=losses, accs=accs,precisions=precisions, recalls=recalls))

    torch.save(net.state_dict(),"/Users/duan/Desktop/ClustRead/data/S7/model.ckpt")




