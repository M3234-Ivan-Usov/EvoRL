import numpy as np
from scipy.stats import tvar, tmean


class MetricEstimator:
    def __init__(self, ds, stat):
        self.ds, self.stat, self.metrics = ds, stat, [
            self.var_clusters_size,
            self.var_mean_dist_to_cent,
            self.mean_dist_point_to_nearest_cent,
            self.mean_min_dist_cent_to_nearest
        ]

    def var_clusters_size(self, clusters):
        return tvar(list(map(len, clusters)))

    def var_mean_dist_to_cent(self, clusters):
        centroids = self.stat.get_centroids(clusters)
        return tvar(self.stat.mean_dists_to_centroids(clusters, centroids))

    def mean_dist_point_to_nearest_cent(self, clusters):
        centroids = self.stat.get_centroids(clusters)
        dists = self.stat.dists_to_clusters(centroids, major='point')
        for cl_idx, cluster in enumerate(clusters):
            for p_id in cluster:
                del dists[p_id][cl_idx]
        return tmean(list(map(min, dists)))

    def mean_min_dist_cent_to_nearest(self, clusters):
        centroids = self.stat.get_centroids(clusters)
        dists = [[self.stat.dist(centroids[x], centroids[y]) if x != y else float('inf')
                 for y in range(len(clusters))] for x in range(len(clusters))]
        return tmean(list(map(min, dists)))

    def __call__(self, labels):
        return np.array([[metric.__call__(clusters) for metric in self.metrics]
                         for clusters in map(self.stat.split_by_clusters, labels)])
