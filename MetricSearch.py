import numpy as np
import pandas as pd
import datetime

from distances import levenshtein

import fastdtw

from dtaidistance import dtw

from utils import *
from preprocess import *

import nerd

from scipy.spatial.distance import squareform, pdist

import TemporalClusterer

#import umap
import hdbscan
import dCollector
from sklearn.model_selection import ParameterGrid
from IPython import get_ipython

from scipy.spatial.distance import hamming


if __name__ == '__main__':

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'qt')

    param_grid = {'metric': ['hamming']
        # 'jaccard', 'cosine',
        # lambda x, y: fastdtw.fastdtw(x, y, dist=hamming)[0]]#, 'euclidean', 'dice'],
        # lambda x, y :dtw.distance_fast(x.astype(np.double), y.astype(np.double))]
                  }


    params = ParameterGrid(param_grid)

    results = pd.DataFrame(data=ParameterGrid(param_grid))
    results['cluster_count'] = 0
    results['activity_mean'] = 0
    results['good_ratio'] = 0
    results['activity_mean'] = 0
    results['ips_mean'] = 0
    results['ips_count'] = 0

    idx = 0

    days = int(sys.argv[4])
    intervals = list(pd.date_range(sys.argv[2], sys.argv[3]))
    # Load preprocessed files
    print(f"Processing files for {intervals[idx].date().isoformat()} - {intervals[idx + days - 1].date().isoformat()}")
    (df, file_list) = load_files(sys.argv[1], intervals[idx].date().isoformat(),
                                 intervals[idx + days - 1].date().isoformat())

    idx = 0
    for p in params:

        print("Clustering")
        tc = TemporalClusterer.TemporalClusterer(min_events=8, min_activity=0.1, max_activity=0.7,
                                                 dist_threshold=0.1, metric=p['metric'],
                                                 )

        dfc = df.copy()
        dfc['labels'] = tc.fit_transform(dfc, [])

        print("Running post process")
        (clusters, series, score, ipseries) = tc.post_process(dfc, file_list, query_nerd=False)

        x = series.apply(lambda x: x[x > 0].mean(), axis=1)

        results.loc[idx, 'cluster_count'] = len(x)  # num of clusters
        results.loc[idx, 'good_cluster_count'] = len(x[x > 0.85])  # num of good clusters
        results.loc[idx, 'good_ratio'] = len(x[x > 0.85]) / len(x)  # ratio
        results.loc[idx, 'activity_mean'] = x.mean()  # mean of mean activity favors good small clusters, i know
        results.loc[idx, 'ips_mean'] = clusters['size'].mean()  # mean count of ips in cluster
        results.loc[idx, 'ips_count'] = clusters['size'].sum()  # total ips assigned to clusters

        print(results.loc[idx, :])
        idx += 1

    # sns.heatmap(results.iloc[:,1:], cmap='vlag')
    from sklearn import metrics
    print(metrics.silhouette_score(ipseries, tc.features.loc[ipseries.index, 'labels']))

    ipsgood = (lambda x: [item for sublist in x for item in sublist])(clusters.loc[x > 0.85, 'ips'])
    print(metrics.silhouette_score(ipseries.loc[ipsgood, :], tc.features.loc[ipseries.loc[ipsgood, :].index, 'labels']))