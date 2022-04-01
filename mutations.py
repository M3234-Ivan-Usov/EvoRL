import copy

import numpy as np
from scipy.special import softmax
from scipy.stats import tmean


class Mutator:
    def __init__(self, ds, stat, elite=0.1):
        self.ds, self.stat = ds, stat
        self.elite = int(len(ds) * elite) + 1
        self.mutations = [
            self.hyper_split, self.merge, self.delete, self.move, self.expand
        ]

    def hyper_split(self, clusters):
        centroids = self.stat.get_centroids(clusters)
        dists = self.stat.mean_dists_to_centoids(clusters, centroids)
        split_idx = np.random.choice(len(clusters), p=softmax(dists))

        spliterator, centroid = clusters[split_idx], centroids[split_idx]
        split_dists = np.array([self.stat.dist(self.ds[idx], centroid) for idx in spliterator])
        the_farthest = self.ds[spliterator[split_dists.argmax()]]

        new_clusters = copy.deepcopy(clusters)
        new_clusters.append(list())
        new_spliterator = new_clusters[split_idx]
        for idx, p in enumerate(new_spliterator):
            if self.stat.dist(self.ds[p], centroid) > self.stat.dist(self.ds[p], the_farthest):
                del new_spliterator[idx]
                new_clusters[-1].append(p)
        return new_clusters

    def merge(self, clusters):
        centroids = self.stat.get_centroids(clusters)
        dists = [self.stat.dist(centroids[x], centroids[y]) for x in range(1, len(clusters)) for y in range(x)]
        rates = list(map(lambda d: 1.0 / (1.0 + d), dists))



    def delete(self, clusters):
        centroids = self.stat.get_centroids(clusters)
        dists = self.stat.dists_to_clusters(centroids)
        nearest = len(self.ds) - self.elite
        sorts = [list(sorted(cl_dists))[:nearest] for cl_dists in dists]
        rates = [tmean(map(lambda s: s / sort_dists[-1], sort_dists)) for sort_dists in sorts]
        deleter_idx = np.random.choice(len(clusters), p=softmax(rates))

    def move(self, labels):
        pass

    def expand(self, labels):
        pass

    def __call__(self, labels, arms):
        return [self.mutations[idx].__call__(labels[idx]) for idx in range(len(arms))]
