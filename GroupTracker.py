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


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    sph.dendrogram(linkage_matrix, **kwargs)


if __name__ == '__main__':

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'qt')

    dfrom = sys.argv[1]#'2020-03-12'
    dto = sys.argv[2]#'2020-03-18'

    results = pd.date_range(dfrom, dto, freq='1D').to_series()\
        .apply(datetime.datetime.strftime, format='./data/%Y-%m-%d_%Y-%m-%d').apply(utils.load_results)

    for index, value in results.iteritems():
        results[index]=results[index].assign(day=index)

    days=len(results)

    X = pd.concat(results.values).fillna(0)
    clustCols = list(X.columns)
    clustCols.remove('day')
    X = X.reset_index()

    clustering = sc.AgglomerativeClustering(affinity='jaccard', linkage='complete', distance_threshold=np.float(sys.argv[3]),
                                            n_clusters=None).fit(X.loc[:, clustCols])

    X['group'] = clustering.labels_

    # print member ips of surviving groups
    X[clustCols + ['group']].groupby('group')\
        .agg(sum)\
        .apply(lambda x: (X.columns.to_series().loc[clustCols].loc[x > 0]).to_list(), axis=1)

    # print composition of clusters by days
    evolution=X[clustCols+['group','day']].groupby(['group', 'day'])\
        .agg(sum).apply(lambda x: (X.columns.to_series().loc[clustCols].loc[x > 0]).to_list(), axis=1).unstack('day')

    # print histogram of surviving days for clusters
    survival = X[['day', 'group', 'labels']].groupby('group')\
        .agg(survival=('group', 'count'), days=('day', list), labels=('labels', list))\
        .sort_values(by='survival', ascending=False)

    ax = survival.loc[survival['survival'] > 1, 'survival'].hist(bins=list(range(1, days+2)), align='left')
    ax.set_title('Group reoccurrence at time range {} - {}'.format(dfrom, dto))
    ax.set_xlabel('times reoccurence')
    ax.set_ylabel('count')


    twoMore=len(survival.loc[survival['survival'] > 2, :])
    oneMore=len(survival.loc[survival['survival'] > 1, :])

    print(('From {} groups, there is {} that did reoccurre and {} that reocurred more than twice.\n' +
          'Ratio for more than two reoccurrences is {}.')
          .format(len(survival), oneMore, twoMore, twoMore/oneMore))

    evolution = evolution.loc[survival.index, :]
    #special func needed to avoid proplem with replacing nans
    counts = evolution.applymap(lambda x: len(x) if x is not np.nan else 0)

    plt.figure()
    counts_norm = (counts.T / counts.T.max()).T
    ax = sns.heatmap(data=counts_norm.loc[survival['survival'] > 1, :])
    ax.set_xticklabels(counts.columns.to_series().apply(datetime.date.isoformat))
