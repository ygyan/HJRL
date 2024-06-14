import torch.nn as nn
from layer import GraphConvolution_, GraphConvolution
import torch.nn.functional as F
import torch

class HJRL(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, activation, neg_slope, X_nodes, Y_nodes):
        super(HJRL, self).__init__()

        self.gc1 = GraphConvolution_(nfeat, nhid, 1, activation, neg_slope)
        self.gc2 = GraphConvolution(nhid, nclass, 1)

        self.dropout = dropout
        self.activation = activation
        self.neg_slope = neg_slope
        self.X_nodes = X_nodes
        self.Y_nodes = Y_nodes

    def forward(self, x, y, HHT, H, HT, HTH):
        HHT_1 = self.gc1(x, HHT)
        H_1 = self.gc1(y, H)
        HT_1 = self.gc1(x, HT)
        HTH_1 = self.gc1(y, HTH)
        HHT_1 = F.dropout(HHT_1, self.dropout, training=self.training)
        H_1 = F.dropout(H_1, self.dropout, training=self.training)
        HT_1 = F.dropout(HT_1, self.dropout, training=self.training)
        HTH_1 = F.dropout(HTH_1, self.dropout, training=self.training)

        x = HHT_1 + H_1
        y = HT_1 + HTH_1
        HHT_2 = self.gc2(x, HHT)
        H_2 = self.gc2(y, H)
        HT_2 = self.gc2(x, HT)
        HTH_2 = self.gc2(y, HTH)
        HHT_2 = F.leaky_relu(HHT_2, negative_slope=self.neg_slope)
        H_2 = F.leaky_relu(H_2, negative_slope=self.neg_slope)
        HT_2 = F.leaky_relu(HT_2, negative_slope=self.neg_slope)
        HTH_2 = F.leaky_relu(HTH_2, negative_slope=self.neg_slope)

        x = HHT_2 + H_2
        y = HT_2 + HTH_2
        reconstruct = torch.cat([x, y], dim=0)

        x_output = F.log_softmax(x, dim=1)
        y_output = torch.sigmoid(y)

        return x_output, y_output, reconstruct