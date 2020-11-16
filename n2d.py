import argparse
import os
import random as rn
import datetime
import time

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch import optim

import umap
from sklearn import metrics
from sklearn import mixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.utils.linear_assignment_ import linear_assignment

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except BaseException:
    print("Missing MulticoreTSNE package.. Only important if evaluating other manifold learners.")

np.set_printoptions(threshold=sys.maxsize)
matplotlib.use('agg')


# Using nn.ModuleList
class Autoencoder(nn.Module):
    def __init__(self, numLayers, encoders=False):
        super().__init__()
        self.layers = nn.ModuleList()
        if encoders:
            for i in range(len(numLayers) - 2):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[-2], numLayers[-1]))
        else:
            for i in range(len(numLayers) - 2):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i+1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[-2], numLayers[-1]))
            for i in range(len(numLayers) - 1, 1, -1):
                self.layers.append(nn.Linear(numLayers[i], numLayers[i-1]))
                self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(numLayers[1], numLayers[0]))

    def forward(self, x):
        y = x
        for i in range(len(self.layers)):
            y = self.layers[i](y)
        
        return y


def eval_other_methods(x, y, names=None):

    gmm = mixture.GaussianMixture(
        covariance_type='full',
        n_components=args.n_clusters,
        random_state=0)
    gmm.fit(x)
    y_pred_prob = gmm.predict_proba(x)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | GMM clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    y_pred = KMeans(
        n_clusters=args.n_clusters,
        random_state=0).fit_predict(x)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | K-Means clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    sc = SpectralClustering(
        n_clusters=args.n_clusters,
        random_state=0,
        affinity='nearest_neighbors')
    y_pred = sc.fit_predict(x)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | Spectral Clustering on raw data")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    if args.manifold_learner == 'UMAP':
        md = float(args.umap_min_dist)
        hle = umap.UMAP(
            random_state=0,
            metric=args.umap_metric,
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors,
            min_dist=md).fit_transform(x)
    elif args.manifold_learner == 'LLE':
        from sklearn.manifold import LocallyLinearEmbedding
        hle = LocallyLinearEmbedding(
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors).fit_transform(x)
    elif args.manifold_learner == 'tSNE':
        method = 'exact'
        hle = TSNE(
            n_components=args.umap_dim,
            n_jobs=16,
            random_state=0,
            verbose=0).fit_transform(x)
    elif args.manifold_learner == 'isomap':
        hle = Isomap(
            n_components=args.umap_dim,
            n_neighbors=5,
        ).fit_transform(x)

    gmm = mixture.GaussianMixture(
        covariance_type='full',
        n_components=args.n_clusters,
        random_state=0)
    gmm.fit(hle)
    y_pred_prob = gmm.predict_proba(hle)
    y_pred = y_pred_prob.argmax(1)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | GMM clustering on " +
          str(args.manifold_learner) + " embedding")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    if args.visualize:
        plot(hle, y, 'UMAP', names)
        y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
        plot(hle, y_pred_viz, 'UMAP-predicted', names)

        return

    y_pred = KMeans(
        n_clusters=args.n_clusters,
        random_state=0).fit_predict(hle)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | K-Means " +
          str(args.manifold_learner) + " embedding")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    sc = SpectralClustering(
        n_clusters=args.n_clusters,
        random_state=0,
        affinity='nearest_neighbors')
    y_pred = sc.fit_predict(hle)
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | Spectral Clustering on " +
          str(args.manifold_learner) + " embedding")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)
    

def cluster_manifold_in_embedding(hl, y, label_names=None):
    # find manifold on autoencoded embedding
    if args.manifold_learner == 'UMAP':
        hl = hl.cpu().data.numpy()
        md = float(args.umap_min_dist)
        hle = umap.UMAP(
            random_state=0,
            metric=args.umap_metric,
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors,
            min_dist=md).fit_transform(hl)
    elif args.manifold_learner == 'LLE':
        hle = LocallyLinearEmbedding(
            n_components=args.umap_dim,
            n_neighbors=args.umap_neighbors).fit_transform(hl)
    elif args.manifold_learner == 'tSNE':
        hle = TSNE(
            n_components=args.umap_dim,
            n_jobs=16,
            random_state=0,
            verbose=0).fit_transform(hl)
    elif args.manifold_learner == 'isomap':
        hle = Isomap(
            n_components=args.umap_dim,
            n_neighbors=5,
        ).fit_transform(hl)

    # clustering on new manifold of autoencoded embedding
    if args.cluster == 'GMM':
        gmm = mixture.GaussianMixture(
            covariance_type='full',
            n_components=args.n_clusters,
            random_state=0)
        gmm.fit(hle)
        y_pred_prob = gmm.predict_proba(hle)
        y_pred = y_pred_prob.argmax(1)
    elif args.cluster == 'KM':
        km = KMeans(
            init='k-means++',
            n_clusters=args.n_clusters,
            random_state=0,
            n_init=20)
        y_pred = km.fit_predict(hle)
    elif args.cluster == 'SC':
        sc = SpectralClustering(
            n_clusters=args.n_clusters,
            random_state=0,
            affinity='nearest_neighbors')
        y_pred = sc.fit_predict(hle)

    y_pred = np.asarray(y_pred)
    # y_pred = y_pred.reshape(len(y_pred), )
    y = y.cpu().data.numpy()
    y = np.asarray(y)
    # y = y.reshape(len(y), )
    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print(args.dataset + " | " + args.manifold_learner +
          " on autoencoded embedding with " + args.cluster + " - N2D")
    print('=' * 80)
    print(acc)
    print(nmi)
    print(ari)
    print('=' * 80)

    if args.visualize:
        plot(hle, y, 'n2d', label_names)
        y_pred_viz, _, _ = best_cluster_fit(y, y_pred)
        plot(hle, y_pred_viz, 'n2d-predicted', label_names)

    return y_pred, acc, nmi, ari


def best_cluster_fit(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    best_fit = []
    for i in range(y_pred.size):
        for j in range(len(ind)):
            if ind[j][0] == y_pred[i]:
                best_fit.append(ind[j][1])
    return best_fit, ind, w


def cluster_acc(y_true, y_pred):
    _, ind, w = best_cluster_fit(y_true, y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def plot(x, y, plot_id, names=None):
    viz_df = pd.DataFrame(data=x[:5000])
    viz_df['Label'] = y[:5000]
    if names is not None:
        viz_df['Label'] = viz_df['Label'].map(names)

    viz_df.to_csv(args.save_dir + '/' + args.dataset + '.csv')
    plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=0, y=1, hue=viz_df.Label.tolist(), legend='full', hue_order=sorted(viz_df['Label'].unique()),
                    palette=sns.color_palette("hls", n_colors=args.n_clusters),
                    alpha=.5,
                    data=viz_df)
    l = plt.legend(bbox_to_anchor=(-.1, 1.00, 1.1, .5), loc="lower left", markerfirst=True,
                   mode="expand", borderaxespad=0, ncol=args.n_clusters + 1, handletextpad=0.01, )

    l.texts[0].set_text("")
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(args.save_dir + '/' + args.dataset +
                '-' + plot_id + '.png', dpi=300)
    plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='(Not Too) Deep',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset', default='mnist', )
    parser.add_argument('gpu', default=0, )
    parser.add_argument('--n_clusters', default=10, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=1000, type=int)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/n2d')
    parser.add_argument('--umap_dim', default=2, type=int)
    parser.add_argument('--umap_neighbors', default=10, type=int)
    parser.add_argument('--umap_min_dist', default="0.00", type=str)
    parser.add_argument('--umap_metric', default='euclidean', type=str)
    parser.add_argument('--cluster', default='GMM', type=str)
    parser.add_argument('--eval_all', default=False, action='store_true')
    parser.add_argument('--manifold_learner', default='UMAP', type=str)
    parser.add_argument('--visualize', default=True, action='store_true')
    args = parser.parse_args()
    print(args)

    label_names = None
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])
    trainset = torchvision.datasets.MNIST(root='/home/mayur/Desktop/Pytorch/data', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=16)
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    net = Autoencoder(numLayers=[trainset.data[0].view(-1).shape[0], 500, 500, 2000, 10])
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.to(device)

    def train(model, optimizer, loss_fn, train_loader, n_epochs, device):
	    for epoch in range(n_epochs):
	        loss_train = 0.0
	        for imgs, labels in train_loader:
	            imgs = imgs.to(device=device)
	            imgs = imgs.view(imgs.shape[0], -1)
	            outputs = model(imgs)
	            loss = loss_fn(outputs, imgs)
	            loss_train += loss.item()
	            optimizer.zero_grad()
	            loss.backward()
	            optimizer.step()
	        if (epoch+1) % 100 == 0:
	                print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch+1, loss_train / len(train_loader)))
	    
	    return model

    pretrain_time = time.time()
    # Pretrain autoencoders before clustering
    if args.ae_weights is None:
        trained_model = train(net, optimizer, loss_fn = nn.MSELoss(), train_loader = trainloader, n_epochs=args.pretrain_epochs, device=device)
        if os.path.isdir('weights'):
            torch.save(trained_model.state_dict(), "weights/"+args.dataset+"-"+str(args.pretrain_epochs)+"-"+"n2d.pt")
        elif os.mkdir('weights'):
            torch.save(trained_model.state_dict(), "weights/"+args.dataset+"-"+str(args.pretrain_epochs)+"-"+"n2d.pt")
        
        pretrain_time = time.time() - pretrain_time
        print("Time to train the autoencoder: " + str(pretrain_time))
    else:
        net.load_state_dict(torch.load('weights/'+args.ae_weights, map_location=device))

    encoder = nn.Sequential(*[net.layers[i] for i in range(7)])
    encoder.to(device)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.save_dir + '/args.txt', 'w') as f:
        f.write("\n".join(sys.argv))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=16)
    for imgs, labels in trainloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        imgs = imgs.view(imgs.shape[0], -1)
        hl = encoder(imgs)
        if args.eval_all:
            eval_other_methods(x, y, label_names)
        clusters, t_acc, t_nmi, t_ari = cluster_manifold_in_embedding(hl, labels, label_names)
        np.savetxt(args.save_dir + "/" + args.dataset + '-clusters.txt', clusters, fmt='%i', delimiter=',')