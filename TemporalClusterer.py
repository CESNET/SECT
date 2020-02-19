import numpy as np
import pandas as pd
import datetime
from utils import *

from preprocess import *

from scipy.spatial.distance import squareform, pdist

#import umap
import hdbscan

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
        self.vect = pd.DataFrame()
        self.pairwise = pd.DataFrame()

    def filter(self, x):
        # Put here means to clean data
        return x.loc[(x['active'] > self.min_events & #minimal activity
                        (x['active'] <= np.ceil(self.max_activity*self.vect_len)) & #maximal activity
                        (x['blocks'] >= self.min_events)) # pattern requirements
                        #(data['count'] >= self.min_events)
                        , :]

    def transform(self, df, labels):

        df.timestamp = np.floor((df.timestamp - df.timestamp.min())/self.aggregation)

        data = df.groupby('ip')['timestamp'].agg([list, 'count'])

        data = data.loc[data['count'] >= self.min_events, :]

        #min_t = min(data['list'].min())
        self.vect_len = df.timestamp.max() + 1

        #self.vect_len = np.int(np.ceil(max_t/self.aggregation))

        # might use expand right away instead of stacking
        data['series'] = data.list.apply(get_bin_series, args=[self.vect_len])
        data['active'] = data.series.apply(np.sum)
        data['blocks'] = data.series.apply(count_blocks)


        return filter(data)

    def fit_transform(self, x, y):

        data = self.transform(x, y)

        # filter
        data = data.loc[(data['active'] > self.min_events &
                        (data['active'] <= np.ceil(self.max_activity*self.vect_len)) &
                        (data['blocks'] >= self.min_events))
                        #(data['count'] >= self.min_events)
                        , :]

        labels = pd.Series([])

        limit = 50000
        if len(data) > 0:
            # stack of features with ip in time index
            if len(data) <= limit:
                self.vect = pd.DataFrame(index=data.index, data=np.stack(data.series))
            else:
                data = data.sample(limit)
                self.vect = pd.DataFrame(index=data.index, data=np.stack(data.series))


            pairwise = pd.DataFrame(squareform(pdist(self.vect, self.metric)), index=self.vect.index,
                                    columns=self.vect.index,
                                dtype=np.float16)

            # prune features/distance matrix
            if self.prune:
                matches = pairwise.apply(lambda x: np.sum(x < self.dist_threshold), raw=True)\
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

        clusters = x['ip'].apply(lambda c: labels[c] if c in labels.index else -1)

        return clusters

