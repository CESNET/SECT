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
        print(f"Processing files for {intervals[idx]} - {intervals[idx+days-1]}")
        (df, file_list) = load_files(sys.argv[1], intervals[idx].date().isoformat(), intervals[idx+days-1].date().isoformat())

        print("Clustering")
        tc = TemporalClusterer.TemporalClusterer(min_events=sys.argv[5], max_activity=sys.argv[6],
                                                 dist_threshold=sys.argv[7],)
        df['labels'] = tc.fit_transform(df, [])

        print("Running post process")
        (clusters, series, score) = tc.post_process(df, file_list)

        # Ranking of clusters, to pick what to focus on
        top10 = clusters.sort_values(by=['score', 'size'], ascending=False).head(10)

        intervals = sample_intervals(series, file_list[0], tc.aggregation) # tc.aggregations should be same as with series

        # only if you want flows and more data
        df_flows = pd.Series(dtype=object)
        df_nerd = pd.DataFrame()

        if sys.argv[8] == 'True':
            df_flows = clusters_get_flows(top10['ips'], intervals.loc[top10.index])

            df_ip = df.loc[df['labels'] > -1, ['ip', 'labels']].loc[df['labels'] > -1]
            df_ip = df_ip.groupby('ip').agg(cluster=('labels', min))
            nerdC = nerd.NerdC()
            df_nerd = nerdC.ip_req(df_ip.index.values)

            df_nerd['cluster'] = df_ip
            del df_ip

            df_flow_views = flows_get_views(df_flows)

            store_analysis(f'./data/{file_list[0]}_{file_list[-1]}/', df.loc[df['labels']>-1, :], clusters, series,
                       df_nerd, df_flows, df_flow_views[0], df_flow_views[1])

