import argparse
import os
import pickle
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from layers import *


class VLTransformer(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num
        
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            #encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)
            
            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout = self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout)
        
    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())
        
        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)
            #x = x.transpose(1, 0)#nn.multi_head_attn
            #x, attn = self.Encoder[i](x, x, x)#nn.multi_head_attn
            #x = x.transpose(1, 0)#nn.multi_head_attn
            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())
           
        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        output, hidden = self.Outputlayer(x, attn_embedding)
        return output, hidden, attn_map
    
class VLTransformer2(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer2, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num

        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []

        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            # encoder = nn.MultiheadAttention(self.d_k * self.n_head, self.n_head, dropout = self.dropout) #nn.multi_head_attn
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)

            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head, dropout=self.dropout)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        d_in = self.d_v * self.n_head * self.modal_num
        self.Outputlayer = OutputLayer2(d_in, self.d_v * self.n_head, self.n_class, self.modal_num, self.dropout,dataname=hyperpm.datname)

    def forward(self, x):
        bs = x.size(0)
        attn_map = []
        x, _attn = self.InputLayer(x)
        attn = _attn.mean(dim=1)
        attn_map.append(attn.detach().cpu().numpy())

        for i in range(self.n_layer):
            x, _attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn = _attn.mean(dim=1)

            x = self.FeedForward[i](x)
            attn_map.append(attn.detach().cpu().numpy())

        x = x.view(bs, -1)
        attn_embedding = attn.view(bs, -1)
        # output, hidden = self.Outputlayer(x, attn_embedding)
        # return output, hidden, attn_map
        output_x, output_attn, hidden = self.Outputlayer(x, attn_embedding)  # hidden (598,72)
        return output_x, output_attn, hidden, attn_map

class VLTransformer_Gate(nn.Module):
    def __init__(self, input_data_dims, hyperpm):
        super(VLTransformer_Gate, self).__init__()
        self.hyperpm = hyperpm
        self.input_data_dims = input_data_dims
        self.d_q = hyperpm.n_hidden
        self.d_k = hyperpm.n_hidden
        self.d_v = hyperpm.n_hidden
        self.n_head = hyperpm.n_head
        self.dropout = hyperpm.dropout
        self.n_layer = hyperpm.nlayer
        self.modal_num = hyperpm.nmodal
        self.n_class = hyperpm.nclass
        self.d_out = self.d_v * self.n_head * self.modal_num
        
        self.InputLayer = VariLengthInputLayer(self.input_data_dims, self.d_k, self.d_v, self.n_head, self.dropout)
        self.Encoder = []
        self.FeedForward = []
        
        for i in range(self.n_layer):
            encoder = EncodeLayer(self.d_k * self.n_head, self.d_k, self.d_v, self.n_head, self.dropout)
            self.add_module('encode_%d' % i, encoder)
            self.Encoder.append(encoder)
            
            feedforward = FeedForwardLayer(self.d_v * self.n_head, self.d_v * self.n_head)
            self.add_module('feed_%d' % i, feedforward)
            self.FeedForward.append(feedforward)
        
        self.FGLayer = FusionGate(self.modal_num)    
        self.Outputlayer = OutputLayer(self.d_v * self.n_head, self.d_v * self.n_head, self.n_class,modal_num=self.modal_num)
        
    def forward(self, x):
        attn_map = []
        bs = x.size(0)
        x, attn = self.InputLayer(x)
        for i in range(self.n_layer):
            x, attn = self.Encoder[i](q=x, k=x, v=x, modal_num=self.modal_num)
            attn_ = attn.mean(dim=1)
            x = self.FeedForward[i](x)
            attn_map.append(attn_.detach().cpu().numpy())
        x, norm = self.FGLayer(x)    
        x = x.sum(-2)/norm
        x = x.view(bs, -1)
        attn_ = attn_.view(bs, -1)
        output, hidden = self.Outputlayer(x,attn_)
        return output, hidden,attn_map
def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix

def _normalize_adj_m(indices, adj_size):
    adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
    row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
    col_sum = 1e-7 + torch.sparse.sum(adj.t(), -1).to_dense()
    r_inv_sqrt = torch.pow(row_sum, -0.5)
    rows_inv_sqrt = r_inv_sqrt[indices[0]]
    c_inv_sqrt = torch.pow(col_sum, -0.5)
    cols_inv_sqrt = c_inv_sqrt[indices[1]]
    values = rows_inv_sqrt * cols_inv_sqrt
    return values

def get_edge_info(x,dev,adj,dropout_rate):
    indices = adj.nonzero()
    rows = indices[:, 0]
    cols = indices[:, 1]

    edges = torch.stack([rows, cols]).type(torch.LongTensor)

    values = _normalize_adj_m(indices=edges, adj_size=torch.Size((x.size(0), x.size(0))))

    degree_len = int(values.size(0) * (1. - dropout_rate))
    degree_idx = torch.multinomial(values, degree_len)
    # random sample
    keep_indices = edges[:, degree_idx]

    masked_adj = torch.sparse.FloatTensor(keep_indices, torch.ones_like(keep_indices[0]), torch.Size((x.size(0), x.size(0)))).to(dev)

    return masked_adj
class GraphLearn(nn.Module):
    def __init__(self, input_dim,input_dim_origin, th, mode='Sigmoid-like',hyperpm=None,input_data_dims = 0):
        super(GraphLearn, self).__init__()
        self.mode = mode
        self.w = nn.Linear(input_dim, 1)
        self.w_origin = nn.Linear(input_dim_origin, 1)
        self.t = nn.Parameter(torch.ones(1))
        self.p = nn.Linear(input_dim, input_dim)
        self.p_origin = nn.Linear(input_dim_origin, input_dim_origin)
        self.threshold = nn.Parameter(torch.zeros(1))
        self.th = th

        self.dims = input_data_dims
        self.weight_ABIDE = nn.Parameter(torch.Tensor([0.25, 0.25, 0.25, 0.25]))
        self.dataname = hyperpm.datname
        self.kNum = hyperpm.kNum

        self.freeze = hyperpm.freeze


        self.origin_adj_file = './graph/{}_origin_adj.pt'.format(self.dataname)
        self.fused_adj_file = './graph/{}_fused_adj.pt'.format(self.dataname)
    def forward(self, x,x_origin,staus="eval"):
        initial_x = x.clone()
        num, feat_dim = x.size(0), x.size(1)

        use_cuda = torch.cuda.is_available()
        dev = torch.device('cuda' if use_cuda else 'cpu')
        
        if self.mode == "Sigmoid-like":
            x = x.repeat_interleave(num, dim = 0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = diff.pow(2).sum(dim=2).pow(1/2)
            diff = (diff + self.threshold) * self.t
            output = 1 - torch.sigmoid(diff)
            
        elif self.mode == "adaptive-learning":
            x = x.repeat_interleave(num, dim = 0)
            x = x.view(num, num, feat_dim)
            diff = abs(x - initial_x)
            diff = F.relu(self.w(diff)).view(num, num)
            output = F.softmax(diff, dim = 1)
        
        elif self.mode == 'weighted-cosine':
            th = self.th

            x = self.p(x)
            x_norm = F.normalize(x,dim=-1)
            score = torch.matmul(x_norm, x_norm.T)

            x_origin = self.p_origin(x_origin)
            x_norm_origin = F.normalize(x_origin, dim=-1)
            # score_origin = torch.matmul(x_norm_origin, x_norm_origin.T)

            modal_num = len(self.dims)
            temp_dim = 0
            matrix_size = x_origin.shape[0]
            score_origin = torch.zeros((matrix_size, matrix_size)).to(dev)
            score_origins = []
            for i in range(modal_num):
                data = x_norm_origin[:, temp_dim: temp_dim + self.dims[i]]
                graph = torch.matmul(data, data.T)
                score_origins.append(graph)
                temp_dim += self.dims[i]
            if self.dataname == "ABIDE":
                score_origin = 0.28 * score_origins[0] + 0.2 * score_origins[1] + 0.17 * score_origins[2] + 0.5 * score_origins[3]

            else:
                score_origin = 0.17* score_origins[0] +0.5* score_origins[1] + 0.14 * score_origins[2] + 0.16 * score_origins[3]+ 0.9* score_origins[4]+ 0.15 * score_origins[5]

            if staus == "train":
                score_origin_masked_adj = get_edge_info(x, dev, score_origin, 0.2)
                score_origin_masked_adj = score_origin_masked_adj.to_dense()
                score_origin = score_origin * score_origin_masked_adj

                if self.dataname == "ABIDE":
                    score_masked_adj = get_edge_info(x, dev, score, 0.1)
                    score_masked_adj = score_masked_adj.to_dense()
                    score = score * score_masked_adj


            scores = 0.9 * score + 0.1* score_origin
            # scores = score


            output = build_knn_neighbourhood(scores, topk=self.kNum)

            # mask = (scores > th).detach().float()
            # markoff_value = 0
            # output = scores * mask + markoff_value * (1 - mask)
        return output,scores

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x1 = F.relu(self.gc1(x, adj))
        x2 = F.dropout(x1, self.dropout, training=self.training)
        x3 = self.gc2(x2, adj)
        return F.log_softmax(x3, dim=1), x2
    
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttConv(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttConv(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1), x