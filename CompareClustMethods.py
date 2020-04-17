# Here should be code to compare results of analysis for different clustering methods
# Intention is to take found clusters and compare them with pretty relaxed metric

import sys

import pandas as pd
import numpy as np

import utils
import datetime
import hdbscan
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.cluster as sc
import scipy.cluster.hierarchy as sph

import scipy.spatial.distance as sd

from IPython import get_ipython


if __name__ == '__main__':

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'qt')

    dfrom = sys.argv[2]#'2020-03-12'
    dto = sys.argv[3]#'2020-03-18'
    path =  sys.argv[1]
    threshold = np.float(sys.argv[4])
    min_occurence = np.int(sys.argv[5])

    hdbscan_groups = pd.read_pickle(f"{path}/hdbscan_{dfrom}_{dto}.pcl")
    agglomerative_groups = pd.read_pickle(f"{path}/agglomerative_{dfrom}_{dto}.pcl")


    tmp = pd.get_dummies(hdbscan_groups.loc[hdbscan_groups['occurence'] >= min_occurence,'ips']\
                         .apply(pd.Series)).assign(method='hdbscan').astype({'method': 'str'})
    tmp.columns = pd.Series(tmp.columns).apply(lambda x: x.split('_')[-1])

    X = pd.get_dummies(agglomerative_groups.loc[agglomerative_groups['occurence'] >= min_occurence, 'ips']\
                       .apply(pd.Series)).assign(method='agglomerative').astype({'method': 'str'})
    X.columns = pd.Series(X.columns).apply(lambda x: x.split('_')[-1])

    tmp = tmp.T.groupby(tmp.columns).agg('max').T
    X = X.T.groupby(X.columns).agg('max').T

    X = pd.concat([X, tmp]).fillna(0)

    X=X.reset_index(drop=True)

    clustCols = X.columns.to_list()
    clustCols.remove('method')


    clustering = sc.AgglomerativeClustering(affinity='jaccard', linkage='complete', distance_threshold=threshold,
                                            n_clusters=None).fit(X[clustCols])
    X['mapping'] =  clustering.labels_

    seenin = X.groupby('mapping').agg(seenin=('method', list))

    print(seenin)

    groups = pd.concat([agglomerative_groups, hdbscan_groups]).reset_index(drop=True)
    groups['method'] = X.method
    groups['mapping'] = X.mapping

    #todo do not merge subclusters
    ipsMethod = groups.groupby(['mapping', 'method']).agg(ips=('ips', lambda x: [item for sublist in x for item in
                                                                            sublist]),colisions=('ips','count')).unstack('method')
    counts = ipsMethod['ips'][:].applymap(lambda x: len(x) if x is not np.nan else 0)

    plt.figure()
    counts_norm = (counts.T / counts.T.max()).T
    ax = sns.heatmap(data=counts_norm, robust=True, annot=counts) # filter here if need be
    #ax.set_xticklabels(counts.columns.to_series().apply(datetime.date.isoformat))
