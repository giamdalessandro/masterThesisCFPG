# Based on https://github.com/tkipf/pygcn/blob/master/pygcn/

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, dropout, log_softmax, nll_loss
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv



class GraphConvolution(nn.Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNSynthetic(nn.Module):
    """3-layer GCN used in GNN Explainer synthetic tasks"""
    def __init__(self, nfeat, nhid, nout, nclass, dropout):
        super(GCNSynthetic, self).__init__()
        self.nclass = nclass

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.gc3 = GraphConvolution(nhid, nout)
        self.lin = nn.Linear(nhid + nhid + nout, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        input_lin = self.embedding(x, adj)
        x = self.lin(input_lin)
        return log_softmax(x, dim=1)

    def embedding(self, x, adj):
        """Computes nodes embeddings, i.e. feed-forward only through the
        convolutional layers of the GCN.
        """
        x1 = relu(self.gc1(x, adj))
        x1 = dropout(x1, self.dropout, training=self.training)
        x2 = relu(self.gc2(x1, adj))
        x2 = dropout(x2, self.dropout, training=self.training)
        x3 = self.gc3(x2, adj)
        input_lin = torch.cat((x1, x2, x3), dim=1)

        return input_lin

    def loss(self, pred, label):
        return nll_loss(pred, label)