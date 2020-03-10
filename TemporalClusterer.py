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


class TemporalClusterer:
    def __init__(self, min_events=2, max_activity=0.8, aggregation=900, min_cluster_size=2, dist_threshold=0.05,
                 metric='jaccard', prune=True, sample=-1):

        # filtering params
        self.min_events = np.int(min_events)
        self.max_activity = np.float(max_activity)

        # clustering params
        self.aggregation = np.int(aggregation)
        self.min_cluster_size = np.int(min_cluster_size)
        self.dist_threshold = np.float(dist_threshold)

        self.metric = metric
        self.prune = prune
        self.sample = sample

        self.vect_len = 0
        self.features = pd.DataFrame()
        self.pairwise = pd.DataFrame()


    def transform(self, df, labels):

        df.timestamp = np.floor((df.timestamp - df.timestamp.min())/self.aggregation)

        data = df.groupby('ip')['timestamp'].agg([list, 'count'])

        data = data.loc[data['count'] >= self.min_events, :]

        #min_t = min(data['list'].min())
        self.vect_len = df.timestamp.max() + 1

        #self.vect_len = np.int(np.ceil(max_t/self.aggregation))

        # might use expand right away instead of stacking
        data['series'] = data.list.apply(get_bin_series, args=[self.vect_len])
        data['activity'] = data.series.apply(np.sum)
        data['blocks'] = data.series.apply(count_blocks)

        data = data.loc[((data['activity'] > self.min_events) &  # minimal activity
                  (data['activity'] <= np.ceil(self.max_activity * self.vect_len)) &  # maximal activity
                  (data['blocks'] >= self.min_events))  # pattern requirements
                # (data['count'] >= self.min_events)
                  , :]

        return data


    def fit_transform(self, x, y):

        data = self.transform(x, y)
        self.features = data

        labels = pd.Series([])

        limit = 50000
        if len(data) > 0:
            # stack of features with ip in time index
            if len(data) <= limit:
                vect = pd.DataFrame(index=data.index, data=np.stack(data.series))
            else:
                #verify this
                data = data.sort_values(inplace=True, ascending=False, by=['activity', 'blocks'])

                vect = pd.DataFrame(index=data.head(limit).index, data=np.stack(data.head(limit).series))
                print(f"Data too big, reduced count from {len(data)} to {limit}", file=sys.stderr)

            pairwise = pd.DataFrame(squareform(pdist(vect, self.metric)), index=vect.index,
                                    columns=vect.index,
                                    dtype=np.float16)

            # prune features/distance matrix
            if self.prune:
                matches = pairwise.apply(lambda x: np.sum(x < self.dist_threshold))\
                    .where(lambda x: x >= self.min_cluster_size)\
                    .dropna()

                self.pairwise = pairwise.loc[matches.index, matches.index]
                # input empty array -1 not decided if so.
            else:
                self.pairwise = pairwise

            if len(self.pairwise) > 1:
                labels = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    metric='precomputed',
                    cluster_sellection_method='leaf'
                ).fit_predict(self.pairwise.astype(np.float)) # why ?

                labels = pd.Series(labels, index=self.pairwise.index)

        self.features['labels'] = labels

        clusters = x['ip'].apply(lambda c: labels[c] if c in labels.index else -1)

        return clusters

    def post_process(self, df, file_list):

        df = df.loc[df['labels'] > -1, :]

        df['type_less'] = df['type'].apply(
            lambda x: x.replace('[', '').replace(']', '').replace(' ', '').replace("'", '').replace('"', '')
                       .replace(', ', '-'))

        df_type = pd.get_dummies(df['type_less'])

        df['origin_less'] = df['origin'].apply(
            lambda x: x.split(',')[-1].rstrip(']').rstrip('}').split(':')[-1].strip(' ').replace('\'', ''))

        df_origin = pd.get_dummies(df['origin_less'])

        clusters_origin = df_origin.groupby(df['labels']).agg('sum')
        clusters_type = df_type.groupby(df['labels']).agg('sum')

        clusters = (df.loc[df.labels > -1]
                    .groupby('labels')
                    .agg(ips=('ip', lambda x: list(set(x))), size=('ip', lambda x: len(set(x))), events=('ip', 'count'),
                         tfrom=('timestamp', min), tto=('timestamp', max),
                         types=('type_less', set),
                         origins=('origin_less', set),
                         )
                    )

        # for filtering
        clusters[['min_blocks', 'min_activity']] = (self.features.groupby('labels').
                                                    agg(min_blocks=('blocks', min), min_activity=('activity', min)))

        #clusters.sort_values(('size'), ascending=False, inplace=True)

        series = (df.loc[df.labels > -1]
                  .groupby(['labels', 'ip'])['timestamp']
                  .agg(list)
                  .apply(lambda x: np.array(get_bin_series(x, self.vect_len), dtype=np.float))
                  .groupby('labels')
                  .agg(list)
                  .apply(lambda x: np.sum(x, axis=0) / len(x))
                  )

        #TODO fix annotation to not use frequency
        series = pd.DataFrame(
            data=np.stack(series),
            index=series.index,
            columns=pd.DatetimeIndex(
                pd.date_range(start=file_list[0], end=file_list[-1]+' 23:59:59.99', periods=self.vect_len+1)[:-1])
                .strftime('%m-%d %H:%M')
        )

        score_sum, score, tags = rank_clusters(clusters, series, clusters_type, clusters_origin)

        clusters['score'] = score_sum
        clusters['tags'] = tags

        return clusters, series


if __name__ == '__main__':

    # Import the necessaries libraries


    # Set notebook mode to work in offline
    # from scipy.cluster.hierarchy import dendrogram, linkage
    #Load preprocessed files
    for day in pd.date_range(sys.argv[2], sys.argv[3]):
        print("Processing files")
        (df, file_list) = load_files(sys.argv[1], day.date().isoformat(), day.date().isoformat())

        print("Clustering")
        tc = TemporalClusterer(min_events=sys.argv[4], max_activity=sys.argv[5], dist_threshold=sys.argv[6])
        df['labels'] = tc.fit_transform(df, [])
        print("Running post process")
        (clusters, series) = tc.post_process(df, file_list)

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

