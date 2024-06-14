import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from utils import load_data, dotdict, seed_everything, accuracy, normalize_sparse_hypergraph_symmetric
import sys
import time
from model import  HJRL
import torch
import torch.nn.functional as F
from torch import nn, optim
import argparse
import csv
import scipy.sparse as sp
import scipy.io as scio



def training(data, args, s=2021):

    seed_everything(seed=s)
    H_trainX = torch.from_numpy(data.H_trainX.toarray()).float().cuda()
    X = torch.from_numpy(data.X.toarray()).float().cuda()
    Y = torch.from_numpy(data.Y.toarray()).float().cuda()
    HHT = torch.from_numpy(data.HHT.toarray()).float().cuda()
    H = torch.from_numpy(data.H.toarray()).float().cuda()
    HT = torch.from_numpy(data.HT.toarray()).float().cuda()
    HTH = torch.from_numpy(data.HTH.toarray()).float().cuda()

    idx_train = torch.LongTensor(data.idx_train).cuda()
    idx_val = torch.LongTensor(data.idx_val).cuda()
    idx_test = torch.LongTensor(data.idx_test).cuda()
    labels_X = torch.LongTensor(np.where(data.labels_X)[1]).cuda()
    labels_Y = torch.LongTensor(data.labels_Y).float().cuda()

    gamma = args.gamma
    epochs = args.epochs
    learning_rate = args.learning_rate
    dropout = args.dropout
    activation = args.activation
    neg_slope = args.neg_slope
    pos_weight_H = float(H_trainX.shape[0] * H_trainX.shape[0] - H_trainX.sum()) / H_trainX.sum()
    model = HJRL(X.shape[1], args.dim_hidden, data.n_class, dropout, activation, neg_slope, data.X_nodes, data.Y_nodes)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    cost_val = []

    for epoch in range(epochs):
        t = time.time()
        model.train()
        x_output, y_output, x = model(X, Y, HHT, H, HT, HTH)
        loss1 = F.nll_loss(x_output[idx_train], labels_X[idx_train])
        hat_x = x[0:data.X_nodes]
        hat_y = x[data.X_nodes : data.X_nodes + data.Y_nodes]
        recovered_H = torch.mm(hat_x, hat_y.t())
        recovered_H = torch.sigmoid(recovered_H)
        loss2 = F.binary_cross_entropy_with_logits(recovered_H, H_trainX, pos_weight=pos_weight_H)
        loss_train = loss1 + gamma * loss2
        acc_train = accuracy(x_output[idx_train], labels_X[idx_train])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if args.verb == 1:
            if epoch % 1 == 0:
                print("Epoch:", '%04d|' % (epoch + 1),
                      "loss1: ", '%f12|' % loss1.item(),
                      "loss2: ", '%f12|' % loss2.item(),
                      "loss: ", '%f12|' % loss_train.item())

        loss_val = F.nll_loss(x_output[idx_val], labels_X[idx_val])
        cost_val.append(loss_val.item())
        acc_val = accuracy(x_output[idx_val], labels_X[idx_val])

        if args.verb2 == 1:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))


        if epoch > args.early_stop and cost_val[-1] > np.mean(cost_val[-(args.early_stop + 1):-1]):
            print("Early stopping...")
            break

    with torch.no_grad():
        model.eval()
        x_output, y_output, x = model(X, Y, HHT, H, HT, HTH)
        loss_test = F.nll_loss(x_output[idx_test], labels_X[idx_test])
        acc_test = accuracy(x_output[idx_test], labels_X[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              "acc_test: {:.4f}".format(acc_test.item()))

    return acc_val.item(), acc_test.item()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--dataname', type=str, nargs='?', default='citeseer1000', help="dataname to run")
    parser.add_argument('--rule', type=int, nargs='?', default=4, help="ablation of rule")
    setting = parser.parse_args()
    rate = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = setting.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '3'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:100'
    device = torch.cuda.current_device()
    h, X, Y, labels_X, labels_Y, idx_train_list, idx_val_list = load_data(setting.dataname)
    H_trainX = h.copy()
    x_n_nodes = X.shape[0]
    y_n_nodes = Y.shape[0]
    HHT, H = normalize_sparse_hypergraph_symmetric(H_trainX)
    HTH, HT = normalize_sparse_hypergraph_symmetric(H_trainX.transpose())

    if setting.dataname == "citeseer1000":
        dim_hidden = 512
        learning_rate = 0.001
        weight_decay = 0.01
        gamma = 10

    elif setting.dataname == "pubmed1000":
        dim_hidden = 512
        learning_rate = 0.01
        weight_decay = 0.001
        gamma = 1

    elif setting.dataname == "cora1000":
        dim_hidden = 512
        learning_rate = 0.001
        weight_decay = 0.0001
        gamma = 1

    elif setting.dataname == "dblp1000":
        dim_hidden = 512
        learning_rate = 0.01
        weight_decay = 0.0001
        gamma = 1

    epochs = 200
    seed = 2021
    early = 100
    dropout = 0
    activation = 'leaky_relu'
    neg_slope = 0.2
    weight = 1
    acc_val = []
    acc_test = []
    for trial in range(100):
        idx_train = idx_train_list[600+trial]
        idx_val = idx_val_list[600+trial]
        idx_test = np.copy(idx_val)

        data = dotdict()
        args = dotdict()
        data.X = X
        data.Y = Y
        data.HHT = HHT
        data.H = H
        data.HT = HT
        data.HTH = HTH
        data.X_nodes = x_n_nodes
        data.Y_nodes = y_n_nodes
        data.H_trainX = H_trainX
        data.labels_X = labels_X
        data.labels_Y = labels_Y
        data.idx_train = idx_train
        data.idx_val = idx_test
        data.idx_test = idx_test
        data.n_class = labels_X.shape[1]

        name = f'{dim_hidden}-{weight_decay}-{learning_rate}-{dropout}-{gamma}'
        print(f'Trial: {trial + 1}, Setting: {name} ...')

        args.dim_hidden = dim_hidden
        args.weight_decay = weight_decay
        args.epochs = epochs
        args.early_stop = early
        args.learning_rate = learning_rate
        args.dropout = dropout
        args.activation = activation
        args.neg_slope = neg_slope
        args.weight = weight
        args.gamma = gamma
        args.verb = 1
        args.verb2 = 1
        val, test = training(data, args, s=seed)
        acc_val.append(val)
        acc_test.append(test)

    acc_val = np.array(acc_val) * 100
    acc_test = np.array(acc_test) * 100
    m_acc = np.mean(acc_test)
    s_acc = np.std(acc_test)
    print("Test set results:", "accuracy: {:.4f}({:.4f})".format(m_acc, s_acc))
