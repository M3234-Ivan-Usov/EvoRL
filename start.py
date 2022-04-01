import pandas as pd
import numpy as np
from numpy.random import randint as rd_int
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score as sil
from sklearn.preprocessing import minmax_scale as norm

from mutations import Mutator
from politician import Politician
from metrics import MetricEstimator
from stats import ClusterStat

LAMBDA, ATTEMPTS, EPOCHS = 8, 200, 2
ds = norm(pd.read_csv('ds.csv'))
stat = ClusterStat(ds)
estimator = MetricEstimator(ds, stat)
mutator = Mutator(ds, stat)

politician = Politician(len(estimator.metrics), len(mutator.mutations))


def initialise():
    labels = [KMeans(rd_int(2, n // 10), init='random', n_init=1, max_iter=10).fit_predict(ds) for _ in range(LAMBDA)]
    return labels, [sil(ds, sample) for sample in labels]


def select(policies):
    return [np.random.choice(len(mutator.mutations), p=policies[sample]) for sample in range(LAMBDA)]


for epoch in range(EPOCHS):
    labels, fitness = initialise()
    for attempt in range(ATTEMPTS):
        metrics = estimator(labels)
        policies = politician(metrics)
        arms = np.array(select(policies.detach().numpy()))
        new_gen = mutator(labels, arms)
        new_fitness = [sil(ds, sample) for sample in new_gen]
        advantages = np.array(new_fitness) - np.array(fitness)
        politician.backprop(policies, arms, advantages)
        population = list(zip(labels + new_gen, fitness + new_fitness))
        population.sort(key=lambda f: f[1], reverse=True)
        print(f"#{attempt}: {population[0][1]}")
        labels, fitness = zip(*population[:LAMBDA])
