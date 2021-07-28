import numpy as np
import torch
import torch.nn as nn
from backup.Feeder import Feeder
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

    np.random.seed(10)
    torch.manual_seed(10)

    writer=SummaryWriter('/Users/duan/Desktop/log')
    losses = AverageMeter()
    accs  = AverageMeter()
    precisions  = AverageMeter()
    recalls  = AverageMeter()


    val_losses=AverageMeter()
    val_accs =AverageMeter()

    trainset = Feeder("/Users/duan/Desktop/ClustRead/data/S4/features.npy",
                      "/Users/duan/Desktop/ClustRead/data/S4/KNN.npy",
                      "/Users/duan/Desktop/ClustRead/data/S4/labels.npy",
                      train=True)
    trainloader = DataLoader(trainset, batch_size=4,shuffle=True)



    # valset=Feeder("/Users/duan/Desktop/ClustRead/data/S6/val/features.npy",
    #                   "/Users/duan/Desktop/ClustRead/data/S6/val/KNN.npy",
    #                   "/Users/duan/Desktop/ClustRead/data/S6/val/labels.npy",
    #                   "/Users/duan/Desktop/ClustRead/data/S6/val/distances.npy",
    #                   train=True)
    # valloader = DataLoader(valset, batch_size=4, shuffle=False)

    # best_val_loss=10000


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
                #在验证集上验证一遍
                # for j, ((val_feat, val_adj, val_dis, val_cid, val_h1id), val_gtmat) in enumerate(valloader):
                #     val_pred = net(feat, adj, dis, h1id)
                #     val_labels = make_labels(gtmat).long()
                #     val_loss = criterion(pred, labels)
                #
                #     val_p, val_r, val_acc = accuracy(pred, labels)  # 计算精确率、召回率、准确率
                #
                #     val_losses.update(val_loss.item(),val_feat.size(0))
                #     val_accs.update(val_acc.item(),val_feat.size(0))
                # if val_losses.avg<best_val_loss:
                #     best_val_loss=val_losses.avg
                #     print("Best Average Val Loss:\t",best_val_loss)
                #     torch.save(net.state_dict(), "/Users/duan/Desktop/ClustRead/data/S6/model.ckpt")
                # val_losses.reset()
                # val_accs.reset()

    #验证最后得到的模型
    # for j, ((val_feat, val_adj, val_dis, val_cid, val_h1id), val_gtmat) in enumerate(valloader):
    #     val_pred = net(feat, adj, dis, h1id)
    #     val_labels = make_labels(gtmat).long()
    #     val_loss = criterion(pred, labels)
    #
    #     val_p, val_r, val_acc = accuracy(pred, labels)  # 计算精确率、召回率、准确率
    #
    #     val_losses.update(val_loss.item(), val_feat.size(0))
    #     val_accs.update(val_acc.item(), val_feat.size(0))
    # if val_losses.avg < best_val_loss:
    #     best_val_loss = val_losses.avg
    #     print("Best Average Val Loss:\t",best_val_loss)
    #     torch.save(net.state_dict(), "/Users/duan/Desktop/ClustRead/data/S6/model.ckpt")


    torch.save(net.state_dict(),"/Users/duan/Desktop/ClustRead/data/S4/model.ckpt")




