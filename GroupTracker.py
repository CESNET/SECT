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


# def group_collect_intel():

def run(argv, min_size):
    #argv[1] - where are data
    #argv[2] - interval
    #argv[3] - freq of analysis
    #4...

    time_window = argv[2].split("_") # '2020-03-12'
    dfrom = time_window[0]
    dto = time_window[-1]  # '2020-03-18'
    freq = argv[4]

    print(f"Loading data from {argv[1]}")
    ips = utils.load_clusters_ips(utils.expand_range(dfrom, dto, freq=freq), argv[1])
    clusters, activity, dfnerd = utils.load_results(utils.expand_range(dfrom, dto, freq=freq), argv[1])



    intervals = len(ips.groupby('interval').agg('count'))

    clustCols = list(ips.columns)
    clustCols.remove('interval')

    # following two are alligned now
    X = ips.reset_index()
    clusters.reset_index(inplace=True)
    activity.reset_index(inplace=True)

    # filter
    filteredips = list(set(utils.list_list_tolist(clusters.loc[clusters['size'] >= min_size, 'ips'])))
    xrowfilter = X.loc[:, filteredips].sum(axis=1) > 0

    # group clusters across time with each other, clusters are disjunct in one day, but might not be across days
    print(f"Finding groups with {argv[5]} tolerance")
    clustering = sc.AgglomerativeClustering(affinity='jaccard', linkage='complete',
                                            distance_threshold=np.float(argv[5]),
                                            n_clusters=None).fit(X.loc[xrowfilter, filteredips])
    X=X.assign(group=-1)
    X.loc[xrowfilter,'group'] = clustering.labels_

    # print member ips of surviving groups
    ipsSuper = X[clustCols + ['group']].groupby('group') \
        .agg(sum) \
        .apply(lambda x: (X.columns.to_series().loc[clustCols].loc[x > 0]).to_list(), axis=1)

    # composition of clusters by days
    series = X[clustCols + ['group', 'interval']].groupby(['group', 'interval']) \
        .agg(sum).apply(lambda x: (X.columns.to_series().loc[clustCols].loc[x > 0]).to_list(), axis=1).unstack(
        'interval')
    series = series.applymap(lambda x: x if x is not np.nan else [])

    # print histogram of surviving days for clusters
    groups = X[['interval', 'group', 'labels']].groupby('group') \
        .agg(occurence=('group', 'count'), intervals=('interval', list), labels=('labels', list)) \
        .sort_values(by='occurence', ascending=False)

    twoMore = len(groups.loc[groups['occurence'] > 2, :])
    oneMore = len(groups.loc[groups['occurence'] > 1, :])

    print(('From {} groups, there is {} that did reoccurre and {} that reocurred more than twice.\n' +
           'Ratio for more than two reoccurrences is {}.')
          .format(len(groups), oneMore, twoMore, twoMore / oneMore))

    # with pd.option_context('display.max_colwidth', None, 'display.max_rows', None):

    groups = groups.join(clusters[['types', 'origins', 'tags']].groupby(X['group']) \
                         .agg(lambda x: list(set([item for sublist in list(x) for item in sublist]))))
    groups = groups.join(clusters[['events', 'min_activity', 'min_blocks', 'score']].groupby(X['group']) \
                         .agg(sum))

    groups['ips'] = ipsSuper
    groups['quality_measure'] = activity.loc[:, map(lambda x: x not in ['labels', 'interval'], list(activity.columns))] \
        .apply(lambda x: np.mean(x[x > 0]), axis=1).groupby(X['group']).agg(min)

    print('Groups that occured more than once:')
    print(groups.loc[groups['occurence'] > 1, :])

    folder = f"{argv[1]}/{argv[3]}/{dfrom}_{dto}_{freq}"
    utils.store_named_df(folder, dict(zip(['groups', 'series'], [groups[groups.index>=0], series[groups.index>=0]])))
    print(f'Results stored in: {folder}')


if __name__ == '__main__':

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'qt')

    run(sys.argv, 3)


