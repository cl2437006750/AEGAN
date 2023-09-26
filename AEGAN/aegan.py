from __future__ import print_function, division
import argparse

from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

from utils import load_data, load_graph
from GNN import GNNLayer
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

        self.self_attn = MultiheadAttention(n_z, n_head)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
        z = self.z_layer(enc_h3)

        z_self_attn, _ = self.self_attn(z, z, z)
        z = z + z_self_attn

        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z


class AEGAN(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(AEGAN, self).__init__()

        # Autoencoder for intra information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        # Cluster layer for autoencoder
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # Cluster layer for GCN
        self.cluster_layer_gcn = Parameter(torch.Tensor(n_clusters, n_clusters))
        torch.nn.init.xavier_normal_(self.cluster_layer_gcn.data)

        # Degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, tra2, tra3, z = self.ae(x)

        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)

        predict = F.softmax(h, dim=1)

        # ---------------------Compute distribution for GCN middle results
        laplacian_kernel_gcn = torch.exp(-torch.sum(torch.abs(h.unsqueeze(1) - self.cluster_layer_gcn), 2) / self.v)
        gaussian_kernel_gcn = 1.0 / (1.0 + torch.sum(torch.pow(h.unsqueeze(1) - self.cluster_layer_gcn, 2), 2) / self.v)
        gaussian_kernel_gcn = gaussian_kernel_gcn.pow((self.v + 1.0) / 2.0)

        partial = 0.1  # Weight coefficient for Laplacian and Gaussian kernels
        weighted_kernel_gcn = partial * laplacian_kernel_gcn + (1 - partial) * gaussian_kernel_gcn

        amplified_kernel_gcn = F.relu(weighted_kernel_gcn * 10)

        k_gcn = (weighted_kernel_gcn.t() / torch.sum(amplified_kernel_gcn, 1)).t()

        # ---------------------

        # ---------------------Compute distribution for autoencoder middle results
        laplacian_kernel = torch.exp(-torch.sum(torch.abs(z.unsqueeze(1) - self.cluster_layer), 2) / self.v)
        gaussian_kernel = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        gaussian_kernel = gaussian_kernel.pow((self.v + 1.0) / 2.0)

        partial = 0.1  # Weight coefficient for Laplacian and Gaussian kernels
        weighted_kernel = partial * laplacian_kernel + (1 - partial) * gaussian_kernel

        amplified_kernel = F.relu(weighted_kernel * 10)

        k = (weighted_kernel.t() / torch.sum(amplified_kernel, 1)).t()

        # ---------------------

        return x_bar, k, predict, z, h, k_gcn


def target_distribution(k):
    weight = k ** 2 / k.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_aegan(dataset):
    model = AEGAN(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0).to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=args.lr)

    adj = load_graph(args.name)
    adj = adj.cuda()

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        _, _, _, _, z = model.ae(data)
    with torch.no_grad():
        _, _, _, _, h, _ = model(data, adj)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    kmeans1 = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred_gcn = kmeans1.fit_predict(h.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    model.cluster_layer_gcn.data = torch.tensor(kmeans1.cluster_centers_).to(device)
    eva(y, y_pred, 'pae')
    eva(y, y_pred_gcn, 'pae_gcn')

    best_acc = float('-inf')
    best_nmi = float('-inf')
    for epoch in range(200):
        if epoch % 1 == 0:
            # Update interval
            _, tmp_k, pred, _, h, k_gcn = model(data, adj)
            tmp_k = tmp_k.data
            q = target_distribution(tmp_k)
            q_gcn = target_distribution(k_gcn)

            delta = 0.9
            q = delta * q + (1 - delta) * q_gcn
            res = pred.data.cpu().numpy().argmax(1)
            current_acc, current_nmi = eva(y, res, epoch + 1)
            if current_acc > best_acc:
                best_acc = current_acc
            if current_nmi > best_nmi:
                best_nmi = current_nmi

        x_bar, k, pred, _, _, k_gcn = model(data, adj)

        kl_loss = F.kl_div(k.log(), q, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), q, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        gcn_loss = F.kl_div(k_gcn.log(), q, reduction='batchmean')

        loss = 0.5 * kl_loss + 0.05 * gcn_loss + re_loss + 0.05 * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    best_acc = best_acc * 100
    best_nmi = best_nmi * 100
    print("The best acc is: {:.2f}".format(best_acc))
    print("The best nmi is: {:.2f}".format(best_nmi))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='cora')
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=30, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("Using CUDA: {}".format(args.cuda))

    device = torch.device("cuda" if args.cuda else "cpu")
    args.pretrain_path = '../data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'cora':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 7
        args.n_input = 1433
        args.n_z = 30

    if args.name == 'citeseer':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703
        args.n_z = 30

    if args.name == 'pubmed':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 3
        args.n_input = 500

    train_aegan(dataset)