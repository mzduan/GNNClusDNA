import numpy as np
import re
from sklearn.metrics import normalized_mutual_info_score
def handle_file(file_name,read_count):

    prediction=np.zeros(read_count,dtype=int)
    f=open(file_name,"r")
    iterator=-1
    while True:
        l=f.readline()
        if l:
            if l[0]=='>':
                iterator=iterator+1
            else:
                l=l.strip("\n")
                read_name=re.split('\s+',l)[2]
                read_id=int(read_name[4:])-1
                prediction[read_id]=iterator
        else:
            break
    f.close()
    return prediction

if __name__ == '__main__':

    #计算true label
    true_label=list()
    iter = 0
    for i in range(1,11):
        f_name="template_"+str(i)+".fa"
        with open("/Users/duan/Desktop/ClustRead/data/S5/S5/"+f_name,"r") as fa:
            while True:
                l=fa.readline()
                if l:
                    if l[0]=='>':
                        true_label.append(iter)
                else:
                    break
            iter=iter+1
    true_label=np.array(true_label,dtype=int)
    read_count=true_label.shape[0]
    predition_label=handle_file("/Users/duan/Desktop/ClustRead/data/MeShClust结果/S5",read_count)
    #计算NMI

    nmi = normalized_mutual_info_score(predition_label, true_label,average_method="arithmetic")
    print("NMI:",nmi)

    #计算纯度:

    #step 1 :将true_label分割成二维数组
    groundtruth=[]
    g=[]
    start=0
    for i in range(true_label.shape[0]):
        if true_label[i]==start:
            g.append(i)
        else:
            groundtruth.append(g)
            start=start+1
            g=[]
            g.append(i)
    groundtruth.append(g)
    #step 2: 将predition_label分割成二维数组
    predition=[]
    p=[]
    start=predition_label[0]
    for i in range(predition_label.shape[0]):
        if predition_label[i] == start:
            p.append(i)
        else:
            predition.append(p)
            start = predition_label[i]
            p = []
            p.append(i)
    predition.append(g)
    #step 3:开始计priurity
    sum_overlap=0
    sum_pred_counts=0
    for i in range(len(predition)):
        sum_pred_counts=sum_pred_counts+len(predition[i])


    for i in range(len(predition)):
        current_pred=predition[i]
        current_max_overlap=0
        for j in range(len(groundtruth)):
            current_true=groundtruth[j]
            overlap=[val for val in current_pred if val in current_true]
            current_max_overlap=current_max_overlap if current_max_overlap>len(overlap) else len(overlap)
        sum_overlap=sum_overlap+current_max_overlap
    purity=sum_overlap/sum_pred_counts
    print("Purity:",purity)

    #step 4:开始计算completeness
    sum_overlap=0
    for i in range(len(groundtruth)):
        current_true=groundtruth[i]
        current_max_overlap=0
        for j in range(len(predition)):
            current_pred=predition[j]
            overlap=[val for val in current_true if val in current_pred]
            current_max_overlap=current_max_overlap if current_max_overlap>len(overlap) else len(overlap)
            # print("Cluster",i,"->Cluster",j,":\t",len(overlap),"/",len(current_true))
        sum_overlap=sum_overlap+current_max_overlap
    completeness=sum_overlap/read_count
    print("Completeness:",completeness)



