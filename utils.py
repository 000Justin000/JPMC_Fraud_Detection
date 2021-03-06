import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta


class MLP(nn.Module):

    def __init__(self, dim_in, dim_out, dim_hidden=20, num_hidden=0, activation=nn.ReLU(), dropout_p=0.0):
        super(MLP, self).__init__()

        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden >= 1:
            self.linears = nn.ModuleList()
            self.linears.append(nn.Linear(dim_in, dim_hidden))
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden-1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise Exception('number of hidden layers must be positive')

        self.activation = activation
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(self.dropout(x)))

        return self.linears[-1](self.dropout(x))


def rand_split(x, ps):
    assert abs(sum(ps) - 1) < 1.0e-10

    shuffled_x = np.random.permutation(x)
    n = len(shuffled_x)
    pr = lambda p: int(np.ceil(p*n))

    cs = np.cumsum([0] + ps)

    return tuple(shuffled_x[pr(cs[i]):pr(cs[i+1])] for i in range(len(ps)))


def time_split(n, ps):
    assert abs(sum(ps) - 1) < 1.0e-10

    idx = np.arange(n)
    pr = lambda p: int(np.ceil(p*n))

    cs = np.cumsum([0] + ps)

    return tuple(idx[pr(cs[i]):pr(cs[i+1])] for i in range(len(ps)))


def load_jpmc_fraud():

    def time_encoding(dt_str):
        dt = datetime.fromisoformat(dt_str)
        soy = datetime(dt.year, 1, 1, 0, 0, 0)
        som = datetime(dt.year, dt.month, 1, 0, 0, 0)
        sow = datetime(dt.year, dt.month, dt.day, 0, 0, 0) - timedelta(days=dt.weekday())
        sod = datetime(dt.year, dt.month, dt.day, 0, 0, 0)
        tfy = (dt - soy) / timedelta(days=366)
        tfm = (dt - som) / timedelta(days=31)
        tfw = (dt - sow) / timedelta(days=7)
        tfd = (dt - sod) / timedelta(days=1)
        tfs = np.array([tfy, tfm, tfw, tfd])
        emb = np.concatenate((np.sin(2 * np.pi * tfs), np.cos(2 * np.pi * tfs)))
        return emb

    dat = pd.read_csv('jpmc_synthetic.csv', delimiter=',', skipinitialspace=True)

    # get time embedding
    time_emb = torch.tensor(dat['Time_step'].apply(time_encoding).array)

    # get logarithmic transformed USD amount
    log_usd = torch.tensor(np.log(dat['USD_Amount']).array)

    # get one-hot encoding for sender country and bene country
    all_countries = pd.concat((dat['Sender_Country'], dat['Bene_Country'])).unique()
    cty2cid = {country: cid for (cid, country) in enumerate(all_countries)}
    sender_cid = torch.tensor(list(dat['Sender_Country'].apply(lambda x: cty2cid[x])))
    bene_cid = torch.tensor(list(dat['Bene_Country'].apply(lambda x: cty2cid[x])))
    sender_cty_emb = F.one_hot(sender_cid, len(all_countries))
    bene_cty_emb = F.one_hot(bene_cid, len(all_countries))

    # get one-hot encoding for sector
    all_sid = dat['Sector'].unique()
    all_sid.sort()
    assert np.all(all_sid == np.arange(len(all_sid)))
    sid = torch.tensor(dat['Sector'])
    sct_emb = F.one_hot(sid, len(all_sid))

    # get one-hot encoding for transaction type
    all_tst = dat['Transaction_Type'].unique()
    tst2tid = {transaction: tid for (tid, transaction) in enumerate(all_tst)}
    tid = torch.tensor(list(dat['Transaction_Type'].apply(lambda x: tst2tid[x])))
    tst_emb = F.one_hot(tid, len(all_tst))

    feature = torch.cat((time_emb, log_usd.view(-1,1), sender_cty_emb, bene_cty_emb, sct_emb, tst_emb), dim=-1)
    label = torch.tensor(dat['Label'].array)

    return feature.numpy(), label.numpy(), dat[['Sender_Id', 'Bene_Id']]


def pad_sequence(sequences, batch_first=False, padding_value=0.0):

    trailing_dims = sequences[0].shape[1:]
    max_len = max([s.size(0) for s in sequences])
    mask_dims = (len(sequences), max_len) if batch_first else (max_len, len(sequences))
    out_dims = mask_dims + trailing_dims

    mask_tensor = torch.zeros(mask_dims, dtype=torch.bool)
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            mask_tensor[i, :length] = True
            out_tensor[i, :length, ...] = tensor
        else:
            mask_tensor[:length, i] = True
            out_tensor[:length, i, ...] = tensor

    return out_tensor, mask_tensor


def preprocess_rnn_jpmc_fraud(feature, label, info):
    info['index'] = info.index
    group = info[['index', 'Sender_Id']].groupby('Sender_Id')['index'].apply(np.array).reset_index(name='indices')

    group_feature = [torch.tensor(feature[idx], dtype=torch.float32) for idx in group['indices']]
    group_label = [torch.tensor(label[idx], dtype=torch.int64) for idx in group['indices']]
    
    feature_padded, feature_mask = pad_sequence(group_feature, batch_first=True)
    label_padded, label_mask = pad_sequence(group_label, batch_first=True)
    assert torch.all(feature_mask == label_mask)

    return feature_padded, label_padded, feature_mask
