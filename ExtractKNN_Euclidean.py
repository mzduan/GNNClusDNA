import os
import math
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KDTree
import re
import random
import parasail
class Extract:
    def __init__(self,f_dirpath):
        self.f_dirpath=f_dirpath
        self.read_list=list()
        self.label_list=list()
        self.read_name_list=list()
        self.k=4
        self.max_read_len=0
        self.item_count=256
        self.chr2int={'A':0,'C':1,'G':2,'T':3}


        self.train_number=0
        fnames=os.listdir(self.f_dirpath)

        self.matrix=parasail.Matrix("/Users/duan/Desktop/matrix.txt")
        iter=0
        for f in fnames:
            if f.endswith('.fa'):
                with open(self.f_dirpath+f) as fa:
                    full_read=''
                    while True:
                        l=fa.readline()
                        if l:
                            l=l.strip('\n')
                            if l[0]=='>':
                                if full_read!="":
                                    self.max_read_len = self.max_read_len if self.max_read_len > len(full_read) else len(full_read)
                                    self.read_list.append(full_read.upper())
                                    full_read=""
                                self.read_name_list.append(l)
                                self.label_list.append(iter)
                            else:
                                full_read+=l
                        else:
                            self.max_read_len = self.max_read_len if self.max_read_len > len(full_read) else len(full_read)
                            self.read_list.append(full_read.upper())
                            break
                    iter = iter+1


        self.label_list=np.array(self.label_list)
        # self.k=round(math.log(self.max_read_len,4)-1)
        self.item_count=int(math.pow(4,self.k))
        self.read_number=len(self.read_list)

        #选择训练数据量
        self.train_number=1000
        self.train_index=None
        self.train_labels=None


        self.histogram = np.zeros((self.read_number, self.item_count))
        self.pos_sum=np.zeros((self.read_number, self.item_count))
        self.neighbors_one=100
        # self.mul_distances = np.zeros((self.read_number, self.read_number, 5))
        #计算非加权histogram
        for i in range(0,self.read_number):
            self.histogram[i],self.pos_sum[i]=self.getHistogram(self.read_list[i])

    def getHistogram(self,read):
        histogram=np.zeros((self.item_count),dtype=int)
        pos_sum=np.zeros((self.item_count),dtype=int)
        temp_str=read[0:self.k]
        index=self.getSubscript(temp_str)
        histogram[index] += 1
        histogram[index]+=0
        for i in range(self.k,len(read)):
            temp_str=temp_str[1:]+read[i]
            index=self.getSubscript(temp_str)
            histogram[index] += 1
            pos_sum[index] +=i-3;
        return (histogram,pos_sum)

    # def getDistance(self,h1,h2,l1,l2):
    #     # feature 1
    #     LD=abs(l1-l2)
    #
    #     # feature 2
    #     Czekanowski=0
    #
    #     # feature 3
    #     Manhattan = 0
    #     for i in range(0,self.item_count):
    #         tmp_min=h1[i] if h1[i]<=h2[i] else h2[i]
    #         tmp_sum=h1[i]+h2[i]
    #         Czekanowski += tmp_min/tmp_sum if tmp_sum!=0 else 0
    #         Manhattan+=abs(h1[i]-h2[i])
    #
    #     # feature 4
    #     h1_mean=np.mean(h1)
    #     h2_mean=np.mean(h2)
    #
    #
    #     up=0
    #     down1=0
    #     down2=0
    #     for i in range(0,self.item_count):
    #         up+=(h1[i]-h1_mean)*(h2[i]-h2_mean)
    #         down1+=pow(h1[i]-h1_mean,2)
    #         down2+=pow(h2[i]-h2_mean,2)
    #     Pearson=up/(pow(down1,0.5)*pow(down2,0.5))
    #
    #
    #     # feature 5
    #     min_sum=0
    #     for i in range(0, self.item_count):
    #         min_sum += h1[i] if h1[i]<h2[i] else h2[i]
    #     Kulzcynski=pow(4,self.k)*(h1_mean+h2_mean)/(2*h1_mean*h2_mean)*min_sum
    #
    #     return [LD,Czekanowski,Manhattan,Pearson,Kulzcynski]
    #     # return Czekanowski


    def getSubscript(self,kmer):
        subscript=0
        for i in range(0,len(kmer)):
            subscript=subscript + (self.chr2int[kmer[i]]-self.chr2int['A'])*math.pow(4,len(kmer)-i-1)
        return int(subscript)
    # def getDistanceMatrix(self):
    #     distances = np.zeros((self.read_number,self.read_number))
    #
    #     for i in range(0,self.read_number):
    #         for j in range(i+1,self.read_number):
    #             h1=self.histogram[i]
    #             h2=self.histogram[j]
    #             l1=len(self.read_list[i])
    #             l2=len(self.read_list[j])
    #             dis=self.getDistance(h1,h2,l1,l2)
    #             distances[i][j]=dis[1]
    #             distances[j][i]=dis[1]
    #             self.mul_distances[i][j]=dis
    #             self.mul_distances[j][i]=dis
    #         if i%100==0:
    #             print("getDistanceMatrix:\tExecute\t",i,"th\tread")
    #
    #     #将mul_distances归一化
    #     for i in range(5):
    #         min_f=np.min(self.mul_distances[:,:,i])
    #         max_f=np.max(self.mul_distances[:,:,i])
    #         self.mul_distances[:,:,i] = (self.mul_distances[:,:,i] - min_f) / (max_f - min_f)
    #
    #
    #     return distances
        # features=np.array(distances)
        # scaler=preprocessing.MinMaxScaler()
        # features=scaler.fit_transform(features)

    def getKNN(self):
        #基于histogram构建kd-tree，使用欧式距离
        k_nearst = np.zeros((self.read_number, self.neighbors_one))
        tree = KDTree(self.histogram, leaf_size=2)
        for i in range(self.read_number):
            dist, ind = tree.query(self.histogram[i,:].reshape(1,-1), k=self.neighbors_one+1)                # doctest: +SKIP
            ind=ind[0,1:]
            k_nearst[i]=ind
            if i%100==0:
                print("Get KNN:\tIterator\t",i)
        k_nearst = k_nearst.astype(np.int64)


        #随机选择 self.train_number 个中心，计算它到它KNN的距离
        self.train_index=random.sample(range(0,self.read_number),self.train_number)
        self.train_KNN=k_nearst[self.train_index]
        self.train_labels=np.zeros((self.train_number,self.neighbors_one),dtype=int)
        row=0
        for i in self.train_index:
            if row%10==0:
                print("Alignment:\tIterator\t", row)
            center_read=self.read_list[i]
            neighbours=k_nearst[i]
            column=0
            for j in neighbours:
                target_read=self.read_list[j]
                score=self.getIdentityScore(center_read,target_read)
                self.train_labels[row][column]=1 if score>=0.66 else 0
                column=column+1
            row=row+1




        return k_nearst
    def getFeatureMatrix(self):
        # 特征一 ： histogram
        features=self.histogram
        # 特征二 ： 各kmer所在read位置的总和
        features = np.concatenate((features,self.pos_sum),axis=1)
        scaler = preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)
        return features

    def getIdentityScore(self,read1,read2):
        aln = parasail.nw_stats(read1,read2,4,2,self.matrix)
        return aln.matches/aln.length
if __name__ == '__main__':
    # print("Get Training Data Features")
    # e=Extract('/Users/duan/Desktop/ClustRead/data/S6/train/')
    # k_nearst=e.getKNN()
    # features=e.getFeatureMatrix()
    # np.save('/Users/duan/Desktop/ClustRead/data/S6/train/features.npy',features)
    # np.save('/Users/duan/Desktop/ClustRead/data/S6/train/KNN.npy', k_nearst)
    # np.save('/Users/duan/Desktop/ClustRead/data/S6/train/labels.npy', e.label_list)
    # np.save('/Users/duan/Desktop/ClustRead/data/S6/train/distances.npy', e.mul_distances)
    #
    #
    # print("Get Validation Data Features")
    # e=Extract('/Users/duan/Desktop/ClustRead/data/S6/val/')
    # k_nearst=e.getKNN()
    # features=e.getFeatureMatrix()
    # np.save('/Users/duan/Desktop/ClustRead/data/S6/val/features.npy',features)
    # np.save('/Users/duan/Desktop/ClustRead/data/S6/val/KNN.npy', k_nearst)
    # np.save('/Users/duan/Desktop/ClustRead/data/S6/val/labels.npy', e.label_list)
    # np.save('/Users/duan/Desktop/ClustRead/data/S6/val/distances.npy', e.mul_distances)


    print("Get Data Features")
    e=Extract('/Users/duan/Desktop/ClustRead/data/S7/S7/')
    k_nearst=e.getKNN()
    features=e.getFeatureMatrix()
    np.save('/Users/duan/Desktop/ClustRead/data/S7/features.npy',features)
    np.save('/Users/duan/Desktop/ClustRead/data/S7/KNN.npy', k_nearst)
    np.save('/Users/duan/Desktop/ClustRead/data/S7/labels.npy', e.label_list)
    np.save('/Users/duan/Desktop/ClustRead/data/S7/train_index.npy', e.train_index)
    np.save('/Users/duan/Desktop/ClustRead/data/S7/train_KNN.npy', e.train_KNN)
    np.save('/Users/duan/Desktop/ClustRead/data/S7/train_label.npy', e.train_labels)





