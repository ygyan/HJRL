import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math

class GraphConvolution_(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, weight, activation, neg_slope, bias=False):
        super(GraphConvolution_, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.neg_slope = neg_slope
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        if weight == 1:
            self.reset_parameters()
        elif weight == 2:
            self.init_uniform_parameters(in_features, out_features)
        elif weight == 3:
            self.init_randn_parameters(in_features, out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def init_uniform_parameters(self, n_in, n_out):
        # = n_in
        n = (n_in + n_out) * 0.5
        # cale = math.sqrt(6/n)
        scale = np.sqrt(3 / n)
        self.weight.data.uniform_(-scale, scale)

    def init_randn_parameters(self, n_in, n_out):
        n = (n_in + n_out) * 0.5
        stdev = np.sqrt(2 / n)
        self.weight.data.normal_(n_in, n_out) * stdev

    def forward(self, input_features, adj):
        support = SparseMM.apply(input_features, self.weight)
        output = SparseMM.apply(adj, support)

        if self.bias is not None:
            output = output + self.bias
        if self.activation == 'elu':
            output = F.elu(output)
        elif self.activation == 'relu':
            output = F.relu(output)
        elif self.activation == 'leaky_relu':
            output = F.leaky_relu(output, negative_slope=self.neg_slope)
        elif self.activation == 'tanh':
            output = torch.tanh(output)
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(output)
        elif self.activation == 'None':
            output = output
        return output


    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, weight, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        if weight == 1:
            self.reset_parameters()
        elif weight == 2:
            self.init_uniform_parameters(in_features, out_features)
        elif weight == 3:
            self.init_randn_parameters(in_features, out_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def init_uniform_parameters(self, n_in, n_out):
        n = (n_in + n_out) * 0.5
        scale = np.sqrt(3 / n)
        self.weight.data.uniform_(-scale, scale)

    def init_randn_parameters(self, n_in, n_out):
        n = (n_in + n_out) * 0.5
        stdev = np.sqrt(2 / n)
        self.weight.data.normal_(n_in, n_out) * stdev

    def forward(self, input_features, adj):
        support = SparseMM.apply(input_features, self.weight)
        output = SparseMM.apply(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        M1 = M1.float()
        M2 = M2.float()
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g = g.float()
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g = g.float()
            g2 = torch.mm(M1.t(), g)

        return g1, g2