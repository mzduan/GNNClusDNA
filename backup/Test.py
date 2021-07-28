import numpy as np
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from backup.Feeder import Feeder
from torch.utils.data import DataLoader
from GCN import gcn
from backup.Train import accuracy
from backup.Train import make_labels
from Meters import AverageMeter
from Vertex import Veretx,connected_components_constraint
from sklearn.metrics import normalized_mutual_info_score
k_at_hop=[30,5]

def load_checkpoint(fpath):
    if osp.isfile(fpath):
        checkpoint = torch.load(fpath,map_location='cpu')
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))

def clusters2labels(clusters, n_nodes):
    labels = (-1)* np.ones((n_nodes,))
    for ci, c in enumerate(clusters):
        for xid in c:
            # print(xid.name,ci)
            labels[xid.name] = ci
    assert np.sum(labels<0) < 1
    return labels


def graph_propagation(edges, score, max_sz, step=0.1, beg_th=0.9, pool=None):
    th=score.min()
    edges = np.sort(edges, axis=1)
    nodes = np.sort(np.unique(edges.flatten()))
    mapping = -1 * np.ones((nodes.max()+1), dtype=np.int)
    mapping[nodes] = np.arange(nodes.shape[0])
    link_idx = mapping[edges]
    vertexs = [Veretx(n) for n in nodes]


    #预处理得分,两次的得分取均值
    score_dict = {}
    for i, e in enumerate(edges):
        if (e[0], e[1]) in score_dict:
            score_dict[e[0], e[1]] = 0.5 * (score_dict[e[0], e[1]] + score[i])
        else:
            score_dict[e[0], e[1]] = score[i]

    #构建图
    for l, s in zip(link_idx, score):
        # print(l,s)
        vertexs[l[0]].add_link(vertexs[l[1]], s)



    iter=1
    # print("Iter:",iter,"\tGenerate Graph...",end="\t")
    comps, remain = connected_components_constraint(vertexs, max_sz)
    # iteration
    components = comps[:]
    # print("Graph Counts:", len(components),"Remain Vertexs:",len(remain))
    while remain:
        th = th + (1 - th) * step
        # print("Iter:", iter, "\tScore Threshold:",th,"\tGenerate Graph...")
        iter=iter+1
        comps, remain = connected_components_constraint(remain, max_sz, score_dict, th)
        components.extend(comps)
        # print("Graph Counts:", len(components),"Remain Vertexs:",len(remain))
    return components


def evaluate(components,groundtruth,true_labels,read_counts):
    # print("Cluster Number:\t",len(components))
    clusters=[]
    for i in range(len(components)):
        clu=list()
        for j in components[i]:
            clu.append(j.name)
        clu.sort()
        clusters.append(clu)
    # print("Prediction:")
    # for i in range(len(components)):
    #     print(len(clusters[i]))
    #     print(clusters[i])


    #忽视个数在5以下的
    cut_clusters=[]
    for i in range(len(components)):
        if len(clusters[i])>=5:
            cut_clusters.append(clusters[i])

    print("Cluster Number(count >=5):\t",len(cut_clusters))
    # for i in range(len(cut_clusters)):
    #     print(len(cut_clusters[i]))
    #     print(cut_clusters[i])

    #计算purity

    pred_clusters=len(clusters)
    true_clusters=len(groundtruth)
    sum_overlap=0

    sum_pred_counts=0
    for i in range(pred_clusters):
        sum_pred_counts=sum_pred_counts+len(clusters[i])


    for i in range(pred_clusters):
        current_pred=clusters[i]
        current_max_overlap=0
        for j in range(true_clusters):
            current_true=groundtruth[j]
            overlap=[val for val in current_pred if val in current_true]
            current_max_overlap=current_max_overlap if current_max_overlap>len(overlap) else len(overlap)
        sum_overlap=sum_overlap+current_max_overlap
    purity=sum_overlap/sum_pred_counts
    print("Purity:",purity)

    #计算NMI
    final_pred = clusters2labels(components, read_counts)
    nmi = normalized_mutual_info_score(final_pred, true_labels,average_method="arithmetic")
    print("NMI:",nmi)


    #计算completeness
    sum_overlap=0
    for i in range(true_clusters):
        current_true=groundtruth[i]
        current_max_overlap=0
        for j in range(pred_clusters):
            current_pred=clusters[j]
            overlap=[val for val in current_true if val in current_pred]
            current_max_overlap=current_max_overlap if current_max_overlap>len(overlap) else len(overlap)
            # print("Cluster",i,"->Cluster",j,":\t",len(overlap),"/",len(current_true))
        sum_overlap=sum_overlap+current_max_overlap
    completeness=sum_overlap/read_counts
    print("Completeness:",completeness)





if __name__ == '__main__':

    #part 1:
    losses = AverageMeter()
    accs  = AverageMeter()
    precisions  = AverageMeter()
    recalls  = AverageMeter()

    ckpt = load_checkpoint("/Users/duan/Desktop/ClustRead/data/S6/model.ckpt")
    net = gcn()
    net.load_state_dict(ckpt)
    net = net.eval()
    testset = Feeder("/Users/duan/Desktop/ClustRead/data/S5/features.npy",
                      "/Users/duan/Desktop/ClustRead/data/S5/KNN.npy",
                      "/Users/duan/Desktop/ClustRead/data/S5/labels.npy"
                      ,train=False)
    testloader = DataLoader(testset, batch_size=4)
    criterion = nn.CrossEntropyLoss()


    edges = list()   #存储每个center 到第一层邻居的边
    scores = list()  #存储对应边的得分

    for i, ((feat, adj,cid, h1id,node_list), gtmat) in enumerate(testloader):

        pred = net(feat, adj,h1id)

        labels = make_labels(gtmat).long()
        loss = criterion(pred, labels)
        pred = F.softmax(pred, dim=1)
        p, r, acc = accuracy(pred, labels)  # 计算精确率、召回率、准确率
        losses.update(loss.item(), feat.size(0))
        accs.update(acc.item(), feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r, feat.size(0))
        # if i % 20 == 0:
        #     print('Loss {losses.val:.3f} ({losses.avg:.3f})\t'
        #           'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
        #           'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
        #           'Recall {recalls.val:.3f} ({recalls.avg:.3f})'
        #           .format(losses=losses, accs=accs, precisions=precisions, recalls=recalls))

        node_list = node_list.long().squeeze().numpy()
        bs = feat.size(0)

        for b in range(bs):
            cidb = cid[b].int().item()
            if bs==1:
                nl = node_list
            else:
                nl = node_list[b]

            for j,n in enumerate(h1id[b]):
                n = n.item()
                edges.append([nl[cidb], nl[n]])
                scores.append(pred[b*30+j,1].item())
    edges = np.asarray(edges)
    scores = np.asarray(scores)

    #part 2:
    # print("Using BFS to Get Clusters")
    components=graph_propagation(edges,scores,30,0.1)

    #part 3:

    #得到groundtruth
    true_labels=np.load("/Users/duan/Desktop/ClustRead/data/S5/labels.npy")
    read_counts=true_labels.shape[0]
    groundtruth=[]
    g=[]
    start=0
    for i in range(true_labels.shape[0]):
        if true_labels[i]==start:
            g.append(i)
        else:
            groundtruth.append(g)
            start=start+1
            g=[]
            g.append(i)
    groundtruth.append(g)

    # print("GroundTruth:")
    # for i in range(len(groundtruth)):
    #     print(len(groundtruth[i]))
    #     print(groundtruth[i])

    # 评估结果

    evaluate(components,groundtruth,true_labels,read_counts)






