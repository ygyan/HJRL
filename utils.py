import numpy as np
import scipy.sparse as sp
import torch
import random
import os
import scipy.io as scio


def load_data(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    data_mat_edge = scio.loadmat("data/{}_edge.mat".format(dataset_str))
    h = data_mat['h']
    X = data_mat['X']
    labels_X = data_mat['labels']
    labels_Y = data_mat_edge['labels']
    idx_train_list = data_mat['idx_train_list']
    idx_val_list = data_mat['idx_val_list']
    Y_node = h.toarray().shape[1]
    H_T = h.toarray().T
    Y = np.zeros((Y_node, X.shape[1]))
    for i in range(Y_node):
        flag = np.where(H_T[i])[0]
        flag = X[flag]
        flag = flag.sum(0)
        Y[i] = flag
    Y = np.where(Y, 1, 0).astype(float)

    X = normalize_features(X)
    Y = normalize_features(sp.csr_matrix(Y))
    return h, X, sp.csr_matrix(Y), labels_X, labels_Y, idx_train_list, idx_val_list

def normalize_sparse_hypergraph_symmetric(H):
    rowsum = np.array(H.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D = sp.diags(r_inv_sqrt)

    colsum = np.array(H.sum(0),dtype='float')
    r_inv_sqrt = np.power(colsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    B = sp.diags(r_inv_sqrt)

    Omega = sp.eye(B.shape[0])

    hx1 = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)
    hx2 = D.dot(H).dot(Omega).dot(B)

    return hx1, hx2

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
