import numpy as np
import pandas as pd
import datetime
from utils import *

from preprocess import *
import nerd

from scipy.spatial.distance import squareform, pdist


#import umap
import hdbscan
import dCollector
import graphing

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from TemporalClusterer import *

if __name__ == '__main__':

    # Import the necessaries libraries


    # Set notebook mode to work in offline
    # from scipy.cluster.hierarchy import dendrogram, linkage
    #Load preprocessed files
    print("Processing files")
    (df, file_list) = load_files(sys.argv[1], sys.argv[2], sys.argv[3])

    print("Clustering")
    tc = TemporalClusterer(min_events=sys.argv[4], max_activity=sys.argv[5], dist_threshold=sys.argv[6])

    tfrom = datetime.datetime.strptime('{} 00:00:00'.format(file_list[0]), timeFormat).timestamp()
    days = (df.timestamp - tfrom)/

    df['labels'] = tc.fit_transform(df, [])
    print("Running post process")
    (clusters, series, score) = tc.post_process(df, file_list)

    #Ranking of clusters, to pick what to focus on
    top10 = clusters.sort_values(by=['score', 'size'], ascending=False).head(10)

    intervals = sample_intervals(series, file_list[0], tc.aggregation) # tc.aggregations should be same as with series

    # only if you want flows and more data
    df_flows = pd.Series(dtype=object)
    df_nerd = pd.DataFrame()

    if sys.argv[7] == 'True':
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


    #%%
    # print(clusters[['events', 'min_activity', 'min_blocks', 'tfrom', 'tto', 'origins', 'types']])
    # clusters[['events', 'min_activity', 'min_blocks', 'tfrom', 'tto', 'origins', 'types']].hist()
    #
    # fig = go.Figure(graphing.genSankey(res.sample(10),['srcip','dstip'],'packets','Sample of packet flows'))
    #
    # fig.show()
    #
    # badIps = clusters.loc[topx.index, 'ips']
    # print(badIps)
    # fig = go.Figure(graphing.genSankey(res.loc[res['srcip'].isin(list(badIps.values[0])),:],['srcip','proto'],
    #                                    'packets', 'Communication by protocols'))
    # fig.show()

