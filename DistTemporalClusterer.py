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


if __name__ == '__main__':

    # Load preprocessed files
    days = int(sys.argv[4])
    intervals = list(pd.date_range(sys.argv[2], sys.argv[3]))


    for idx in range(0, len(intervals), days):
        print(f"Processing files for {intervals[idx].date().isoformat()} - {intervals[idx+days-1].date().isoformat()}")


        (df, file_list) = load_files(argv[1], day.date().isoformat(), day.date().isoformat())

        print("Clustering")
        tc = TemporalClusterer(min_activity=argv[4], max_activity=argv[5], dist_threshold=argv[6],
                               method=argv[8], batch_limit=20000, prune_distmat=True, min_cluster_size=2)
        # prune=argv[8]=='True')
        vect, feat, type, origin = tc.transform(df, day, day + datetime.timedelta(
            days=1))  # not needed to redo if aggregation does not change

        labels = tc.fit_transform(vect, feat)

        print("Running post process")
        (clusters, series, stats, dfnerd) = tc.postprocess(labels,
                                                           vect, feat, type, origin,
                                                           day, day + datetime.timedelta(days=1),
                                                           query_nerd=False,
                                                           third_party_path=f'{argv[1]}/3rdParty')

        store_named_df(f'./{argv[1]}/{argv[3]}/{file_list[0]}_{file_list[-1]}/',
                       dict(zip(['clusters', 'series', 'stats', 'ipseries', 'feature', 'idea', 'detector'],
                                [clusters, series, stats, vect, feat, type, origin])))
        if len(dfnerd) > 0:
            store_named_df(f'./{argv[1]}/{argv[3]}/{file_list[0]}_{file_list[-1]}/',
                           dict(zip(['dfnerd'], [dfnerd])))

    return


