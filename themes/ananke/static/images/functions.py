from torch.linalg import norm
import torch
import numpy as np
import matplotlib.pyplot as plt
from random import randint


def prepare_data(low, high, n_clusters, min_distance=0, plotting=False, block=True):
    """generates clustered simulated data"""
    too_narrow = True
    while too_narrow:
        clusters = np.random.multivariate_normal((0, 0), 2 * n_clusters * np.eye(2), size=n_clusters)
        dists = np.zeros((n_clusters,n_clusters))
        for i in range(n_clusters):
            for j in range(i):
                dists[i,j] = np.linalg.norm(clusters[i,:] - clusters[j,:])
        idx = np.tril_indices(n_clusters, k=-1)
        if np.min(dists[idx])>min_distance:
            too_narrow = False
    d_list = []
    for i in range(clusters.shape[0]):
        c1,c4 = 1 + 0.1*np.random.randn(2)
        c2,c3 = 0.1*np.random.randn(2)
        cor = 0.5*np.array([[c1,c2],[c3,c4]])
        while not (np.all(np.linalg.eigvals(cor) > 0.01)):
            cor += 0.1*np.eye(2)
        n_points = randint(low, high)
        d = np.random.multivariate_normal(clusters[i,:], cor, size=n_points)
        d_list.append(d)

    data = np.vstack(d_list)
    np.random.shuffle(data)
    if plotting:
        fig, ax = plt.subplots()
        for d in d_list:
            ax.scatter(d[:,0], d[:,1], s=2)
        ax.scatter(clusters[:,0], clusters[:,1], s=20)
        plt.show(block = block)

        return data, clusters, ax
    else:
        return data, clusters


def loss_fn(data, centers, alpha=1, beta = 0.5, gamma=0.1):
    # alpha is adjustable parameter. Should be of order of the distance between clusters
    n_centers = centers.shape[0]
    diffs_c = torch.zeros(n_centers)
    for i in range(n_centers):
        diffs = data - centers[i,:]
        kernel = torch.exp(-norm(diffs,axis=1)/alpha)
        norms = norm(diffs, axis=1)**2
        diffs_c[i] = torch.sum(kernel*norms)
    loss1 = torch.sum(diffs_c)
    normalization1 = data.shape[0]*n_centers
    loss1 = loss1/normalization1
    # loss 1 enforces centers to moved into clusters

    loss2 = torch.Tensor([0])
    for i in range(n_centers):
        for j in range(0,i):
            norm_c = norm(centers[i,:] - centers[j,:])**2
            loss2 += 1/norm_c
    normalization2 = n_centers
    loss2 = loss2/normalization2
    # loss2 penalizes centers becoming equal

    center_of_mass = torch.mean(data, dim=0)
    star = data - center_of_mass
    data_radius = torch.max(norm(star,axis=1))
    loss3 = torch.Tensor([0])
    for i in range(n_centers):
        norm_com = norm(centers[i, :] - center_of_mass) ** 2
        loss3 += norm_com
    return beta*loss1 + (1-beta)*loss2 + gamma/data_radius*loss3, loss1


def plot_results(c,d, ax, block=False, clear=True):
    if clear:
        ax.cla()
    ax.scatter(d[:, 0], d[:, 1], s=4)
    ax.scatter(c[:, 0], c[:, 1], s=20)
    plt.show(block=block)
    plt.pause(0.001)


def k_means_step(data, z, w, i, R, C, plotting=False, ax=None):
    """Implementation of k-means algorithm to find clusters of data points"""
    point_added = False
    znew = data[i, :]
    distances = np.linalg.norm(z - znew, axis=1)
    # If the new point is far away from all other points (specified by R), add it to the candidates:
    if np.all(distances > R):
        z = np.vstack((z, znew))
        w = np.append(w, 1)
        point_added = True

    if not point_added:
        # k-means iteration:
        idx = np.argmin(distances)
        xnew = (z[idx, :] * w[idx] + znew) / (w[idx] + 1)
        z[idx, :] = xnew
        w[idx] = w[idx] + 1
        # If two candidate points are too close (specified by C), average them together:
        bad_pts = []
        for k in range(len(z)):
            for j in range(k):
                if np.linalg.norm(z[k, :] - z[j, :]) < C:
                    z[j, :] = (z[j, :] * w[j] + z[k, :] * w[k]) / (w[j] + w[k])
                    w[j] = 0.5 * (w[j] + w[k])
                    bad_pts.append(k)
        z = np.delete(z, bad_pts, 0)
        w = np.delete(w, bad_pts)

    if plotting and i%10==0:
        if not ax:
            raise AttributeError('If plotting==True, axis has to be provided.')
        ax.cla()
        ax.scatter(data[:, 0], data[:, 1], s=5)
        ax.scatter(z[:,0],z[:,1], s=40)
        plt.show(block=False)
        plt.xlim((1.1 * np.min(data[:, 0]), 1.1 * np.max(data[:, 0])))
        plt.ylim((1.1 * np.min(data[:, 1]), 1.1 * np.max(data[:, 1])))
        plt.pause(0.01)

    return z, w

