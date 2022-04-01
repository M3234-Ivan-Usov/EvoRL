import numpy as np


class ClusterStat:
    def __init__(self, ds):
        self.ds = ds

    def dist(self, x, y):
        return np.sum((x - y) ** 2)

    def get_centroids(self, clusters):
        return [sum([self.ds[idx] for idx in cl]) / len(cl) for cl in clusters]

    def split_by_clusters(self, labels):
        lab_max = np.max(labels)
        clusters = [list() for _ in range(lab_max + 1)]
        for idx, sample in enumerate(labels):
            clusters[sample].append(idx)
        return clusters

    def mean_dists_to_centroids(self, clusters, centroids):
        dists_to_centroid = list()
        for cl_idx, centroid in enumerate(centroids):
            cluster = clusters[cl_idx]
            dists = [self.dist(self.ds[idx], centroid) for idx in cluster]
            dists_to_centroid.append(sum(dists) / len(cluster))
        return dists_to_centroid

    def dists_to_clusters(self, centroids, major='cluster'):
        if major == 'cluster':
            return [[self.dist(self.ds[idx], centroid) for idx in range(len(self.ds))] for centroid in centroids]
        elif major == 'point':
            return [[self.dist(self.ds[idx], centroid) for centroid in centroids] for idx in range(len(self.ds))]
        else:
            raise ValueError(f"Expected 'cluster' or 'point', got {major}")