import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
from utils import *


x, y, info = load_jpmc_fraud()
tr_idx, va_idx, te_idx = time_split(x.shape[0], [0.6, 0.2, 0.2])
_, cts = np.unique(y, return_counts=True)
c_weight = cts**-1.0 / (cts**-1.0).sum()

method = 'RNN'

def roc_auc(logP, y):
    if type(logP) == torch.Tensor:
        logP = logP.detach().cpu().numpy()
    if type(y) == torch.Tensor:
        y = y.cpu().numpy()
    return roc_auc_score(y, np.exp(logP[:,1]))
    
if method == 'XGBoost':
    model = XGBClassifier()
    model.fit(x[tr_idx], y[tr_idx], sample_weight=c_weight[y[tr_idx]])
    tr_logP = np.log(model.predict_proba(x[tr_idx]))
    va_logP = np.log(model.predict_proba(x[va_idx]))
    te_logP = np.log(model.predict_proba(x[te_idx]))
    tr_auc = roc_auc(tr_logP, y[tr_idx])
    va_auc = roc_auc(va_logP, y[va_idx])
    te_auc = roc_auc(te_logP, y[te_idx])
    print('train roc_auc: {:5.3}, validation roc_auc: {:5.3f}, test roc_auc: {:5.3f}'.format(tr_auc, va_auc, te_auc))

elif method == 'MLP':
    device = 'cuda'
    
    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.int64).to(device)
    model = nn.Sequential(MLP(x.shape[-1], 2, dim_hidden=256, num_hidden=2, activation=nn.LeakyReLU(), dropout_p=0.3), nn.LogSoftmax(dim=-1)).to(device)
    c_weight = torch.tensor(c_weight, dtype=torch.float32).to(device)
    
    optimizer = torch.optim.AdamW([{'params': model.parameters(), 'lr': 1.0e-3}], weight_decay=2.5e-4)
    va_opt, te_opt = 0.0, 0.0
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        tr_logP = model(x[tr_idx])
        tr_loss = F.nll_loss(tr_logP, y[tr_idx], weight=c_weight)
        tr_loss.backward()
        optimizer.step()
    
        model.eval()
        with torch.no_grad():
            tr_logP = model(x[tr_idx])
            tr_loss = F.nll_loss(tr_logP, y[tr_idx], weight=c_weight)
            tr_auc = roc_auc(tr_logP, y[tr_idx])
            va_logP = model(x[va_idx])
            va_loss = F.nll_loss(va_logP, y[va_idx], weight=c_weight)
            va_auc = roc_auc(va_logP, y[va_idx])
        print('epoch {:3d}, train loss: {:5.3f}, train roc_auc: {:5.3f}, validation loss: {:5.3f}, validation roc_auc: {:5.3f}'.format(epoch, tr_loss, tr_auc, va_loss, va_auc))
    
        if va_auc > va_opt:
            va_opt = va_auc
            with torch.no_grad():
                te_logP = model(x[te_idx])
                te_loss = F.nll_loss(te_logP, y[te_idx], weight=c_weight)
                te_auc = roc_auc(te_logP, y[te_idx])
            te_opt = te_auc
    
    print('optimal test roc_auc: {:5.3f}'.format(te_opt))

elif method == 'RNN':
    device = 'cuda'
    
    x, y, mask = preprocess_rnn_jpmc_fraud(x, y, info)
    x = x.to(device)
    y = y.to(device)
    mask = mask.to(device)

    tr_idx, va_idx, te_idx = rand_split(x.shape[0], [0.6, 0.2, 0.2])
    rnn = nn.LSTM(x.shape[-1], 256, 2, batch_first=True, dropout=0.3).to(device)
    log_p = nn.Sequential(nn.Linear(256, 2), nn.LogSoftmax(dim=-1)).to(device)
    c_weight = torch.tensor(c_weight, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW([{'params': rnn.parameters(), 'lr': 1.0e-3}, {'params': log_p.parameters(), 'lr': 1.0e-3}], weight_decay=2.5e-4)
    n_batch = 10

    def train(idx, split):
        rnn.train()
        log_p.train()
        total_loss = 0.0
        logP_list = []
        y_list = []
        for batch_idx in rand_split(idx, [1.0/n_batch]*n_batch):
            batch_x = x[batch_idx]
            batch_y = y[batch_idx]
            batch_mask = mask[batch_idx]

            optimizer.zero_grad()

            batch_z, _ = rnn(batch_x)
            batch_logP = log_p(batch_z)

            bc_mask = batch_mask.reshape(-1)
            bc_logP = batch_logP.reshape(-1, 2)[bc_mask]
            bc_y = batch_y.reshape(-1)[bc_mask]

            bc_loss = F.nll_loss(bc_logP, bc_y, weight=c_weight)
            bc_loss.backward()

            optimizer.step()

            total_loss += bc_loss.item()
            logP_list.append(bc_logP.detach())
            y_list.append(bc_y)

        print('epoch {:3d}, {} loss: {:5.3f}, {} roc_auc: {:5.3f}'.format(epoch, split, total_loss/n_batch, split, roc_auc(torch.cat(logP_list, dim=0), torch.cat(y_list, dim=0))))

    def evaluation(idx, split):
        rnn.eval()
        log_p.eval()
        total_loss = 0.0
        logP_list = []
        y_list = []
        for batch_idx in rand_split(idx, [1.0/n_batch]*n_batch):
            batch_x = x[batch_idx]
            batch_y = y[batch_idx]
            batch_mask = mask[batch_idx]

            with torch.no_grad():
                batch_z, _ = rnn(batch_x)
                batch_logP = log_p(batch_z)

            bc_mask = batch_mask.reshape(-1)
            bc_logP = batch_logP.reshape(-1, 2)[bc_mask]
            bc_y = batch_y.reshape(-1)[bc_mask]

            bc_loss = F.nll_loss(bc_logP, bc_y, weight=c_weight)
            total_loss += bc_loss.item()
            logP_list.append(bc_logP.detach())
            y_list.append(bc_y)

        print('           {} loss: {:5.3f}, {} roc_auc: {:5.3f}'.format(split, total_loss/n_batch, split, roc_auc(torch.cat(logP_list, dim=0), torch.cat(y_list, dim=0))))

        return roc_auc(torch.cat(logP_list, dim=0), torch.cat(y_list, dim=0))
            
    va_opt, te_opt = 0.0, 0.0
    for epoch in range(30):
        train(tr_idx, 'train')
        va_auc = evaluation(va_idx, '  val')
        if va_opt < va_auc:
            va_opt = va_auc
            te_opt = evaluation(te_idx, ' test')

    print('optimal test roc_auc: {:5.3f}'.format(te_opt))
