import numpy as np
import pandas as pd
import datetime
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

if __name__ == '__main__':

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'qt')

    param_grid = {'min_evetns' : [None, 8],#quantile 75% observed in data, if pruning does not happen limit amount of runs of method
                  'min_activity': [5/96],
                  'max_activity': [91/96],#np.array(range(6, 9, 1))/10,
                  'dist_threshold' : [None, 0.2, 0.1, 0.05, 0.025, 0.0125],#force removal of neighborless entities from pairwise matrix
                  'min_cluster_size': [5],
                  'aggr' : [300, 900, 1800],
                  'method' : ['hdbscan'],
                  'metric' :['jaccard','hamming','cos']
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
        tc = TemporalClusterer.TemporalClusterer(min_events=p['min_events'], max_activity=p['max_activity'],
                                                 dist_threshold=p['dist_threshold'], min_cluster_size=p['min_cluster_size'],
                                                 )

        dfc = df.copy()
        dfc['labels'] = tc.fit_transform(dfc, [])

        print("Running post process")
        (clusters, series, score, ipseries) = tc.post_process(dfc, file_list, query_nerd=False)


        x = series.apply(lambda x: x[x > 0].mean(), axis=1)

        results.loc[idx, 'cluster_count'] = len(x) # num of clusters
        results.loc[idx, 'good_cluster_count'] = len(x[x>0.85]) # num of good clusters
        results.loc[idx, 'good_ratio'] = len(x[x>0.85])/len(x) # ratio
        results.loc[idx, 'activity_mean'] = x.mean() # mean of mean activity favors good small clusters, i know
        results.loc[idx, 'ips_mean'] = clusters['size'].mean() # mean count of ips in cluster
        results.loc[idx, 'ips_count'] = clusters['size'].sum() # total ips assigned to clusters

        print(results.loc[idx, :])
        idx += 1

    sns.heatmap(results.corr(), cmap='vlag')