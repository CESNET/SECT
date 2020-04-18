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
    freq = sys.argv[4]
    paths = [f"{path.strip(' ')}/{dfrom}_{dto}_{freq}" for path in sys.argv[1].split('~')]
    threshold = np.float(sys.argv[5])
    min_occurrence = np.int(sys.argv[6])

    groups_dict = dict()
    tmp = pd.DataFrame()
    X = pd.DataFrame()
    groups_all = pd.DataFrame()
    for path in paths:
        groups_dict[path] = utils.load_named_df(path, ['series', 'groups']) #, 'args'])

        groups = groups_dict[path].get('groups').assign(method=path).astype({'method': 'str'})
        groups = groups.loc[groups['occurence'] > min_occurrence, :] #necessary to filter here already because of following groupby
        tmp = pd.get_dummies(groups['ips'].apply(pd.Series)).assign(method=path).astype({'method': 'str'})
        tmp.columns = pd.Series(tmp.columns).apply(lambda x: x.split('_')[-1])
        # Merge columns with same name/IP
        tmp = tmp.T.groupby(tmp.columns).agg('max').T

        X = pd.concat([X, tmp])
        groups_all = pd.concat([groups_all, groups])

    X = X.fillna(0)

    X = X.reset_index(drop=True)
    groups_all.reset_index(inplace=True)

    #clust_rows = (groups_all['occurence'] >= min_occurrence),

    clust_cols = X.columns.to_list()
    clust_cols.remove('method')

    #clust_cols = utils.get_list_of_dummies(X.loc[clust_rows,clust_cols].max(axis=0))

    X['mapping'] = -1
    clustering = sc.AgglomerativeClustering(affinity='jaccard', linkage='complete', distance_threshold=threshold,
                                            n_clusters=None).fit(X.loc[:, clust_cols])
    X['mapping'] = clustering.labels_
    groups_all['mapping'] = clustering.labels_

    #Printing
    seenin = X.groupby('mapping').agg(seenin=('method', list))

    print(seenin)

    series = groups_all.groupby(['mapping', 'method']).agg(ips=('ips', utils.list_list_tolist), colisions=('ips','count')).unstack('method')

    utils.store_named_df(f'{paths[0]}/../../matches/{dfrom}_{dto}_{freq}', dict(zip(['matches', 'series'],[groups_all, series])))