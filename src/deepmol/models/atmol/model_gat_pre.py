import random

import numpy as np
import torch
import torch.nn as nn
"""
预训练模型
transformer
"""
class GATCon(torch.nn.Module):
    def __init__(self, n_output=128,  num_features_xt=78, output_dim=128, dropout=0.2, encoder1=None, encoder2=None):

        super(GATCon, self).__init__()
        self.num_features_xt = num_features_xt
        self.n_output = n_output
        self.dropout = dropout
        self.output_dim = output_dim

        self.gat1 = encoder1
        self.gat2 = encoder2


        # predict head
        self.pre_head = nn.Sequential(
            nn.Linear(output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, self.n_output)
        )

    def forward(self, data1):
        x, edge_index, batch, x_size, edge_size = data1.x, data1.edge_index,data1.batch, data1.x_size, data1.edge_size

        # encoder graph
        x1, weight1 = self.gat1(x, edge_index, batch)

        # predict graph layers
        out1 = self.pre_head(x1)

        # encoder graph
        x_del, edge_index_del = attention_del(weight1, x, edge_index,edge_size,x_size)

        x2, weight2 = self.gat1(x_del, edge_index_del, batch)

        # predict graph layers
        out2 = self.pre_head(x2)

        len_w = edge_index.shape[1]
        w1 = weight1[1]
        ew1 = w1[:len_w]
        xw1 = w1[len_w:]

        return x1, out1, x2, out2, ew1, xw1


def attention_del(weight1,x2,edge_index,edge_size,x_size):
    device = weight1[1].device
    with torch.no_grad():
        len_w = edge_index.shape[1]
        edge_weight = weight1[1][:len_w]
        edge_weight_fc = torch.mean(edge_weight, dim=1)
        count = 0

        del_weights = torch.LongTensor()
        # if device != "cpu":
        del_weights = del_weights.to(device)

        for size in edge_size:
            if count < len_w:
                weight = edge_weight_fc[count:count+size]
            
            del_indices = get_allIndex(weight, len(weight)) + count
            # del_indices = get_randomIndex(weight,len(weight))+count
            # del_indices = get_rouletteIndex(weight, len(weight)) + count
            del_weights = torch.cat((del_weights, del_indices), 0)
            count = count + size

        del_indices, indices = torch.sort(del_weights, dim=0, descending=True)
        edge_index = del_edge(edge_index,del_indices)


        x_weight = weight1[1][len_w:]
        x_weight_fc = torch.mean(x_weight, dim=1)
        count = 0
        del_weights_x = torch.LongTensor()
        # if device != "cpu":
        del_weights_x = del_weights_x.to(device)
        for size in x_size:
            if count < len_w:
                weight = x_weight_fc[count:count+size]

            del_indices = get_allIndex(weight, len(weight)) + count
            # del_indices = get_randomIndex(weight,len(weight))+count
            # del_indices = get_rouletteIndex(weight, len(weight)) + count
            del_weights_x = torch.cat((del_weights_x, del_indices), 0)
            count = count + size
        del_indices, ind = torch.sort(del_weights_x,descending=True)
        x2 = del_x(x2,del_indices)

    return x2, edge_index

def get_randomIndex(weight,len_w):
    sorted, indices = torch.sort(weight, dim=0, descending=False)
    p = 0.25
    len_p = int(len_w * p)
    all_slice = random.sample(indices.tolist(), len_p)
    del_indices = torch.LongTensor(all_slice)
    del_indices, indices = torch.sort(del_indices, dim=0, descending=True)
    return del_indices.cuda()

def get_rouletteIndex(weight,len_w):
    weight_list = weight.tolist()
    weight_fitness = np.array(weight_list).sum()
    fit_ratio = [i / weight_fitness for i in weight_list]
    fit_ratio_add = [0]  # 个体累计概率
    for i in fit_ratio:
        fit_ratio_add.append(fit_ratio_add[len(fit_ratio_add) - 1] + i)  # 计算每个个体的累计概率，并存放到fit_ratio_add中
    fit_ratio_add = fit_ratio_add[1:]  # 去掉首位的0

    p = 0.25
    len_p = int(len_w * p)
    rand_list = [random.uniform(0, 1) for _ in range(len_p)]
    rand_list.sort()
    all_slice = []
    fit_index,index = 0,0
    while index < len(rand_list) and (fit_index + 1) < len(fit_ratio_add):
        if fit_ratio_add[fit_index] < rand_list[index] < fit_ratio_add[fit_index + 1]:
            all_slice.append(fit_index)
            index = index + 1
        else:
            fit_index = fit_index + 1

    del_indices = torch.LongTensor(all_slice)
    del_indices, indices = torch.sort(del_indices, dim=0, descending=True)
    return del_indices.cuda()


def get_allIndex(weight,len_w):
    device = weight.device
    # max-weight True min-weight False
    sorted, indices = torch.sort(weight, dim=0, descending=True)
    p = 0.2
    len_q = int(len_w * p)
    top_indices, ind = torch.sort(indices[:len_q], descending=True)
    remain_indices, ind = torch.sort(indices[len_q:], descending=True)

    top_slice = random.sample(top_indices.tolist(), int(len_q))
    remain_slice = []
    del_indices = torch.LongTensor(top_slice + remain_slice)
    del_indices, indices = torch.sort(del_indices, dim=0, descending=True)
    # if device != "cpu":
    del_indices = del_indices.to(device)
    
    return del_indices

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

def del_edge(edge_index,del_indices):
    edge_index_tmp1 = edge_index[0]
    edge_index_tmp2 = edge_index[1]
    for ei in del_indices:
        edge_index_tmp = del_tensor_ele(edge_index_tmp1,ei)
        edge_index_tmp1 = edge_index_tmp
        edge_index_tmp = del_tensor_ele(edge_index_tmp2,ei)
        edge_index_tmp2 = edge_index_tmp
    return torch.stack([edge_index_tmp1,edge_index_tmp2])

def del_x(x,del_indices):
    device = x.device
    x_tmp = x
    for ei in del_indices:
        x_zero = torch.LongTensor(np.zeros(len(x_tmp[ei]))).unsqueeze(0)
        x_zero = x_zero.to(device)
        x_t = torch.cat((x_tmp[:ei], x_zero),dim=0)
        x_t = x_t.to(device)
        x_tmp = torch.cat((x_t, x_tmp[ei+1:]),dim=0)

        #    if device != "cpu":
        x_tmp = x_tmp.to(device)
           
    return x_tmp