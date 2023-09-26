import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.nn import Linear
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from evaluation import eva
from torch.nn import Linear, MultiheadAttention

class AE(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_head=1):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        self.self_attn = MultiheadAttention(n_z, n_head)  # 添加自注意力层

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        # 添加自注意力层
        z_self_attn, _ = self.self_attn(z, z, z)
        z = z + z_self_attn

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, z

class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        t = torch.from_numpy(np.array(self.x[idx])).float(), torch.from_numpy(np.array(idx))
        return t


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def pretrain_ae(model, dataset, y):
    train_loader = DataLoader(dataset, batch_size=50, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=1e-3)
    for epoch in range(20):
        # adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.cuda()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).cuda().float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))
            kmeans = KMeans(n_clusters=3, n_init=20).fit(z.data.cpu().numpy())
            eva(y, kmeans.labels_, epoch)

        torch.save(model.state_dict(), 'pubmed_agcn.pkl')


model = AE(
    n_enc_1=500,
    n_enc_2=500,
    n_enc_3=2000,
    n_dec_1=2000,
    n_dec_2=500,
    n_dec_3=500,
    n_input=500,
    n_z=30, ).cuda()

x = np.loadtxt('pubmed.txt', dtype=float)
y = np.loadtxt('pubmed_label.txt', dtype=int)

# for i, dataset in enumerate(LoadDataset(x)):
#     pretrain_ae(model, dataset, y)
dataset = LoadDataset(x)
pretrain_ae(model, dataset, y)