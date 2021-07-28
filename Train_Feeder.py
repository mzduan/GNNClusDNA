import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
class Train_Feeder(data.Dataset):
    def __init__(self, feat_path, knn_graph_path, index_path,train_knn_path,label_path,seed=1, k_at_hop=[100,5], active_connection=5):
        self.features = np.load(feat_path)
        self.knn_graph = np.load(knn_graph_path)[:, :k_at_hop[0]]
        self.indexs=np.load(index_path)
        self.train_knn=np.load(train_knn_path)
        self.labels = np.load(label_path)
        self.num_samples = len(self.indexs)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection



    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        #hops[0]存储了第一层的邻居，hops[1]存储了第二层的邻居
        hops = list()

        center_node = self.indexs[index]
        hops.append(set(self.knn_graph[center_node][0:]))   #set之后 数量可能小于30？



        for d in range(1,self.depth):
            hops.append(set())
            for h in hops[-2]:
                hops[-1].update(set(self.knn_graph[h][0:self.k_at_hop[d]]))

        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([center_node, ])  #所有节点
        unique_nodes_list = list(hops_set)
        unique_nodes_map = {j: i for i, j in enumerate(unique_nodes_list)}

        center_idx = torch.Tensor([unique_nodes_map[center_node],]).type(torch.long)
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        center_feat = torch.Tensor(self.features[center_node]).type(torch.float)
        feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        feat = feat - center_feat

        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(unique_nodes_list)


        A = torch.zeros(num_nodes, num_nodes)


        _, fdim = feat.shape
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)


        for node in unique_nodes_list:
            neighbors = self.knn_graph[node, 0:self.active_connection]
            for n in neighbors:
                if n in unique_nodes_list:
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1
                    # dis[unique_nodes_map[n], unique_nodes_map[node]] = torch.Tensor(self.distances[node][n])
                    # dis[unique_nodes_map[node], unique_nodes_map[n]] = torch.Tensor(self.distances[node][n])

        D = A.sum(1, keepdim=True)
        A = A.div(D)
        A_ = torch.zeros(max_num_nodes, max_num_nodes)
        # dis_ = torch.zeros(max_num_nodes, max_num_nodes,5)
        A_[:num_nodes, :num_nodes] = A
        # dis_[:num_nodes,:num_nodes]=dis

        # labels = self.all_labels[np.asarray(unique_nodes_list)]
        # labels = torch.from_numpy(labels).type(torch.long)

        # print(one_hop_idcs)
        # one_hop_labels = labels[one_hop_idcs]

        #找到one_hop_idcs和labels上位置的对应关系


        # center_label = labels[center_idx]

        # print(one_hop_idcs)
        # print(one_hop_labels)
        # print(center_label)
        # edge_labels_1=(center_label == one_hop_labels).long()

        pos_map=dict()
        for i in range(0,len(one_hop_idcs)):
            pos_map[one_hop_idcs[i].item()]=i




        t_knn=self.train_knn[index]
        t_label=self.labels[index]

        transfer_t_knn=list()
        for k in t_knn:
            transfer_t_knn.append(unique_nodes_map[k])
        # transfer_t_knn=torch.Tensor(transfer_t_knn).type(torch.long)
        edge_labels = np.zeros(len(one_hop_idcs),dtype=int)

        for t in range(len(transfer_t_knn)):
            edge_labels[pos_map[transfer_t_knn[t]]]=t_label[t]


        return (feat, A_, center_idx, one_hop_idcs),edge_labels