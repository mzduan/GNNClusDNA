###################################################################
# File Name: gcn.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Fri 07 Sep 2018 01:16:31 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()
    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x 

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, agg):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(
                torch.FloatTensor(in_dim *2, out_dim))

        #考虑各距离之间的权重
        # self.dis_weight = nn.Parameter(
        #         torch.FloatTensor(5, 1))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        init.xavier_uniform_(self.weight)
        # init.xavier_uniform_(self.dis_weight)
        init.constant_(self.bias, 0)
        self.agg = agg()

    def forward(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)
        agg_feats = self.agg(features,A)
        cat_feats = torch.cat([features, agg_feats], dim=2)  #点的特征，邻接矩阵乘以点的特征得到的特征  1024维


        # dis_feats = torch.einsum('bxnd,df->bxnf', (dis, self.dis_weight))
        # dis_feats = dis_feats.view(dis_feats.shape[0],dis_feats.shape[1], dis_feats.shape[2])   #对五个距离加权求和，weight_表示各距离的权重 1
        # dis_agg_feats=self.agg(features,dis_feats)  #距离矩阵乘以点的特征得到的特征


        # cat_feats_=torch.cat([cat_feats,dis_agg_feats],dim=2)
        out = torch.einsum('bnd,df->bnf', (cat_feats, self.weight))
        out = F.relu(out + self.bias)
        return out 
        

class gcn(nn.Module):
    def __init__(self):
        super(gcn, self).__init__()
        self.bn0 = nn.BatchNorm1d(512, affine=False)
        self.conv1 = GraphConv(512, 512, MeanAggregator)
        self.conv2 = GraphConv(512, 512, MeanAggregator)
        self.conv3 = GraphConv(512, 256, MeanAggregator)
        self.conv4 = GraphConv(256, 256,MeanAggregator)
        
        self.classifier = nn.Sequential(
                            nn.Linear(256, 256),
                            nn.PReLU(256),
                            nn.Linear(256, 2))
    
    def forward(self, x, A,one_hop_idcs, train=True):
        # data normalization l2 -> bn
        B,N,D = x.shape
        x = x.view(-1, D)
        x = self.bn0(x)
        x = x.view(B,N,D)


        x = self.conv1(x,A)
        x = self.conv2(x,A)
        x = self.conv3(x,A)
        x = self.conv4(x,A)
        k1 = one_hop_idcs.size(-1)
        dout = x.size(-1)
        edge_feat = torch.zeros(B,k1,dout)
        for b in range(B):
            edge_feat[b,:,:] = x[b, one_hop_idcs[b]]  
        edge_feat = edge_feat.view(-1,dout)
        pred = self.classifier(edge_feat)
            
        # shape: (B*k1)x2
        return pred





