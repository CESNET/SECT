import numpy as np
import pandas as pd
import datetime
from utils import *

from preprocess import *
import nerd

from scipy.spatial.distance import squareform, pdist
import sklearn.cluster as sc

# import umap
import hdbscan
import dCollector
import graphing

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


class TemporalClusterer:
    def __init__(self, min_events=None, min_activity=0.05, max_activity=0.8, aggregation=900, min_cluster_size=2,
                 dist_threshold=0.05,
                 metric='jaccard', method='hdbscan', sample=-1, batch_limit=40000, prune_distmat=False):

        # filtering params

        self.min_events = min_events
        self.max_activity = np.float(max_activity)
        self.min_activity = np.float(min_activity)

        # clustering params
        self.aggregation = np.int(aggregation)
        self.min_cluster_size = np.int(min_cluster_size)

        self.dist_threshold = None
        self.prune = (prune_distmat == 'True') or prune_distmat is True
        if dist_threshold is not None:
            self.dist_threshold = np.float(dist_threshold)

        self.metric = metric
        self.sample = sample

        self.vect_len = np.int(0)
        self.features = pd.DataFrame()
        self.pairwise = pd.DataFrame()

        self.method = method
        if (['agglomerative', 'hdbscan', 'match', 'dbscan', 'optics'].index(self.method) < 0):
            raise ValueError

        self.limit = np.int(batch_limit)

        self.default_dist_threshold = 0.05

        self.min_blocks = 2

    def series_divide(self, division, series, limit=10000):

        division.loc[:] = 0

        # Start dividing by pattern in activity.
        for x in range(np.ceil(np.log2(len(series.columns))).astype(np.int), 2, -1):
            # print('step is {}'.format(2**x))
            # See if division needs to be done
            z = division.value_counts()
            stopper = True
            for w in z.index:
                # For every group index that has group size larger than limit, an only for those
                if z[w] > limit:
                    # Refine division
                    work_set = series.loc[division == w]
                    ws_div = division.loc[division == w]
                    # Shift value indexes higher, not to overvrite higher previous division result
                    ws_div *= 2 ** x

                    idx = 0

                    for y in range(0, len(series.columns), 2 ** x):
                        ws_div.loc[:] = ((work_set.iloc[:, y:y + 2 ** x].apply(max, axis=1) > 0).astype(np.int) *
                                         (2 ** idx))
                        idx += 1

                    # Update new division
                    division.loc[ws_div.index] = ws_div
                    stopper = False

            if stopper:
                break

        return division

    def transform(self, df, labels):

        #        intervals = np.floor((df.timestamp - df.timestamp.min())/self.aggregation)

        #        data = intervals.groupby(df['ip']).agg([list, 'count'])

        df['timestamp'] = np.floor((df.timestamp - df.timestamp.min()) / self.aggregation)

        data = df['timestamp'].groupby(df['ip']).agg([list, 'count'])

        self.vect_len = np.int(df.timestamp.max() + 1)

        if self.min_events is None:
            self.min_events = np.ceil(self.vect_len * self.min_activity)
        else:
            self.min_events = np.int(self.min_events)

        data = data.loc[data['count'] >= self.min_events, :]

        # min_t = min(data['list'].min())

        # self.vect_len = np.int(np.ceil(max_t/self.aggregation))

        # might use expand right away instead of stacking
        data['series'] = data.list.apply(get_bin_series, args=[self.vect_len])
        data['activity'] = data.series.apply(np.sum)
        data['blocks'] = data.series.apply(count_blocks)

        data = data.loc[((data['activity'] >= self.min_activity * self.vect_len) &  # minimal activity
                         (data['activity'] <= np.ceil(self.max_activity * self.vect_len)))  # &  # maximal activity
               #  (data['blocks'] >= self.min_blocks))  # pattern requirements
               # (data['count'] >= self.min_events)
        , :]

        return data

    def fit_transform(self, x, y):

        dataAll = self.transform(x, y)
        self.features = dataAll

        labelsAll = pd.Series(data=-1, index=dataAll.index)

        # limit  #40000 - 12.8 GiB pairwise matrix size

        if len(dataAll) > 0:

            division = pd.Series(data=0, index=dataAll.index)
            seriesAll = pd.DataFrame(index=dataAll.index, data=np.stack(dataAll['series']))

            if len(dataAll) > self.limit:
                division = self.series_divide(division, seriesAll, limit=self.limit * 0.7)

            z = division.value_counts()

            labels_ofs = 0

            for group in z.index:
                data = dataAll.loc[division == group]
                vect = seriesAll.loc[division == group]
                pairwise = pd.DataFrame
                # stack of features with ip in time index
                if len(data) == 0:
                    continue
                # if group is too big, take head, most active with most blocks or sample...
                if len(data) > self.limit:
                    # data.sort_values(inplace=True, ascending=False, by=['activity', 'blocks'])
                    data.sort_values(inplace=True, ascending=False, by=['count'])

                    vect = vect.loc[data.head(self.limit).index, :]
                    print(f"Data too big, reduced count from {len(data)} to {self.limit}", file=sys.stderr)

                if len(vect) > 1:
                    pairwise = pd.DataFrame(squareform(pdist(vect, self.metric)), index=vect.index,
                                                columns=vect.index,
                                                dtype=np.float16)

                        # prune features/distance matrix

                    if self.dist_threshold is not None and self.prune:
                        matches = pairwise.apply(lambda x: np.sum(x <= self.dist_threshold)) \
                            .where(lambda x: x > self.min_cluster_size) \
                            .dropna()  # there is +1

                        pairwise = pairwise.loc[matches.index, matches.index]
                        # input empty array -1 not decided if so.
                    #else:
                    #    pairwise = pairwise
                    # calculate distance matrix and prepare fro clustering
                    if len(pairwise) > 1:
                        # try agglomerative
                        if self.method == 'hdbscan':
                            labels = hdbscan.HDBSCAN(
                                min_cluster_size=self.min_cluster_size,
                                metric='precomputed',
                                # cluster_selection_method='leaf'
                            ).fit_predict(pairwise.astype(np.float))  # why ?

                            labels = pd.Series(labels + labels_ofs, index=pairwise.index)


                        elif self.method == 'dbscan':
                            eps = self.dist_threshold
                            if self.dist_threshold is None:
                                eps = self.default_dist_threshold
                            labels = sc.DBSCAN(
                                eps=eps, min_samples=self.min_cluster_size,
                                metric='precomputed'
                            ).fit_predict(pairwise.astype(np.float))  # why ?

                            labels = pd.Series(labels + labels_ofs, index=pairwise.index)

                        elif self.method == 'optics':
                            eps = self.dist_threshold
                            if self.dist_threshold is None:
                                eps = self.default_dist_threshold
                            labels = sc.OPTICS(
                                max_eps=eps, #min_samples=self.min_cluster_size,
                                min_cluster_size=self.min_cluster_size,
                                metric='precomputed'
                            ).fit_predict(pairwise.astype(np.float))  # why ?

                            labels = pd.Series(labels + labels_ofs, index=pairwise.index)

                        elif self.method == 'agglomerative':
                            AC = sc.AgglomerativeClustering(
                                affinity='precomputed',
                                linkage='complete',
                                distance_threshold=self.dist_threshold,
                                n_clusters=None
                            )
                            labels = AC.fit_predict(pairwise)  # why ?

                            labels = pd.Series(labels + labels_ofs, index=pairwise.index)
                            # Filter smaller clusters then minimum
                            labels = labels.loc[labels.map(labels.value_counts()) >= self.min_cluster_size]


                        elif self.method == 'match':
                            AC = sc.AgglomerativeClustering(
                                affinity='precomputed',
                                linkage='complete',
                                distance_threshold=0.0001,
                                n_clusters=None
                            )
                            labels = AC.fit_predict(pairwise)  # why ?

                            labels = pd.Series(labels + labels_ofs, index=pairwise.index)
                            # Filter smaller clusters then minimum
                            labels = labels.loc[labels.map(labels.value_counts()) >= self.min_cluster_size]

                            # labels = self.pairwise.groupby(list(range(0, tc.vect_len))) \
                            #    .agg(group_size=('activity', 'count'), activity=('activity', 'min'))

                        labels_ofs = labels.max() + 1
                        labelsAll.loc[labels.index] = labels

        self.features['labels'] = labelsAll

        # takes too long on bigger data, if it becomes troublous, rewrite would be required
        clusters = x['ip'].apply(lambda c: labelsAll[c] if c in labelsAll.index else -1)

        return clusters

    def post_process(self, df, file_list, query_nerd=False):

        dfAll = df


        df = df.assign(type_less=df['type'].apply(
            lambda x: x.replace('[', '').replace(']', '').replace(' ', '').replace("'", '').replace('"', '')
                .replace(', ', '-')))

        df_type = pd.get_dummies(df['type_less'])

        df = df.assign(origin_less=df['origin'].apply(
            lambda x: x.split(',')[-1].rstrip(']').rstrip('}').split(':')[-1].strip(' ').replace('\'', '')))

        df_origin = pd.get_dummies(df['origin_less'])


        clusters_origin = df_origin.groupby(df['labels']).agg('sum')
        clusters_type = df_type.groupby(df['labels']).agg('sum')

        df = df.loc[df['labels'] > -1, :] #!!!

        clusters = (df
                    .groupby('labels')
                    .agg(ips=('ip', lambda x: list(set(x))), size=('ip', lambda x: len(set(x))), events=('ip', 'count'),
                         tfrom=('timestamp', min), tto=('timestamp', max),
                         types=('type_less', set),
                         origins=('origin_less', set),
                         )
                    )

        ipseries = (df
                    .groupby(['ip'])['timestamp']
                    .agg(list)
                    .apply(lambda x: np.array(get_bin_series(x, self.vect_len)).astype(np.float))
                    )

        ipseries = pd.DataFrame(data=np.stack(ipseries), index=ipseries.index, columns=pd.DatetimeIndex(
            pd.date_range(start=file_list[0],
                          end=datetime.datetime.strptime(file_list[-1], '%Y-%m-%d') + datetime.timedelta(days=1),
                          periods=self.vect_len + 1)[:-1])
                                .strftime('%Y-%m-%d %H:%M'))

        # for filtering
        clusters[['min_blocks', 'min_activity']] = (self.features.groupby('labels').
                                                    agg(min_blocks=('blocks', min), min_activity=('activity', min)))

        # clusters.sort_values(('size'), ascending=False, inplace=True)

        series = (df
                  .groupby(['labels', 'ip'])['timestamp']
                  .agg(list)
                  .apply(lambda x: np.array(get_bin_series(x, self.vect_len), dtype=np.float))
                  .groupby('labels')
                  .agg(list)
                  .apply(lambda x: np.sum(x, axis=0) / len(x))
                  )

        series = pd.DataFrame(
            data=np.stack(series),
            index=series.index,
            columns=pd.DatetimeIndex(
                pd.date_range(start=file_list[0],
                              end=datetime.datetime.strptime(file_list[-1], '%Y-%m-%d') + datetime.timedelta(days=1),
                              periods=self.vect_len + 1)[:-1])
                .strftime('%Y-%m-%d %H:%M')
        )

        # (self.elen, self.edescr) = self.eval(series)

        #clusters = clusters.loc[clusters.index>=0] #! necessary
        # drop bad clusters
        valid = series  # .loc[series.apply(lambda x: x[x > 0].mean(), axis=1) > 0.85, :]
        clusters_v = clusters  # .loc[valid.index, :]

        score_sum, score, tags, dfnerd = rank_clusters(clusters_v, valid, clusters_type.loc[valid.index, :],
                                                       clusters_origin.loc[valid.index, :], query_nerd=query_nerd)

        clusters_v['score'] = score_sum
        clusters_v['tags'] = tags

        return clusters_v, valid, score, ipseries, dfnerd

    def eval(self, series):
        x = series.apply(lambda x: x[x > 0].mean(), axis=1)
        return (len(series), x.describe())

    def post_process3p(self, df, file_list, query_nerd=True, third_party_path='./data/3rdParty'):
        
        dfAll = df

        df = df.loc[df['labels'] > -1, :] #!!!

        dfAll = dfAll.assign(type_less=dfAll['type'].apply(
            lambda x: x.replace('[', '').replace(']', '').replace(' ', '').replace("'", '').replace('"', '')
                .replace(', ', '-')))

        type_tags_dummies = pd.get_dummies(dfAll['type_less']).groupby['ip'].agg(sum)

        dfAll = dfAll.assign(origin_less=dfAll['origin'].apply(
            lambda x: x.split(',')[-1].rstrip(']').rstrip('}').split(':')[-1].strip(' ').replace('\'', '')))

        origin_tag_dummies = pd.get_dummies(dfAll['origin_less']).groupby['ip'].agg(sum)


        #clusters_origin = df_origin.groupby(dfAll['labels']).agg('sum')
        #clusters_type = df_type.groupby(dfAll['labels']).agg('sum')

        df = dfAll.loc[dfAll['labels'] > -1, :] #!!!

        clusters = (df
                    .groupby('labels')
                    .agg(ips=('ip', lambda x: list(set(x))), size=('ip', lambda x: len(set(x))), events=('ip', 'count'),
                         tfrom=('timestamp', min), tto=('timestamp', max),
                         types=('type_less', set),
                         origins=('origin_less', set),
                         )
                    )

        ipseries = (df
                    .groupby(['ip'])['timestamp']
                    .agg(list)
                    .apply(lambda x: np.array(get_bin_series(x, self.vect_len)).astype(np.float))
                    )

        ipseries = pd.DataFrame(data=np.stack(ipseries), index=ipseries.index, columns=pd.DatetimeIndex(
            pd.date_range(start=file_list[0],
                          end=datetime.datetime.strptime(file_list[-1], '%Y-%m-%d') + datetime.timedelta(days=1),
                          periods=self.vect_len + 1)[:-1])
                                .strftime('%Y-%m-%d %H:%M'))

        # for filtering
        clusters[['min_blocks', 'min_activity']] = (self.features.groupby('labels').
                                                    agg(min_blocks=('blocks', min), min_activity=('activity', min)))

        # clusters.sort_values(('size'), ascending=False, inplace=True)

        series = (df
                  .groupby(['labels', 'ip'])['timestamp']
                  .agg(list)
                  .apply(lambda x: np.array(get_bin_series(x, self.vect_len), dtype=np.float))
                  .groupby('labels')
                  .agg(list)
                  .apply(lambda x: np.sum(x, axis=0) / len(x))
                  )

        series = pd.DataFrame(
            data=np.stack(series),
            index=series.index,
            columns=pd.DatetimeIndex(
                pd.date_range(start=file_list[0],
                              end=datetime.datetime.strptime(file_list[-1], '%Y-%m-%d') + datetime.timedelta(days=1),
                              periods=self.vect_len + 1)[:-1])
                .strftime('%Y-%m-%d %H:%M')
        )

        clusters = clusters.assign(quality_measure=series.apply(lambda x: np.mean(x[x>0])))

        trd = load_named_df(third_party_path, ['greynoise_ip_db_indication',
                                                      'greynoise_ip_db_lists',
                                                      'mentat_ip_db_indication',
                                                      'mentat_ip_db_lists',
                                                      'nerd_ip_db_indication',
                                                      'nerd_ip_db_lists'])

        greyTags, greyTagOutlier = process_tags(dfAll, clusters, trd['greynoise_ip_db_indication'], 0.25)
        grey = pd.DataFrame([greyTags, greyTagOutlier]).T
        grey.columns=['grey_tags', 'grey_outlier_measure']

        ideaTags, ideaTagOutlier = process_tags(dfAll, clusters, type_tags_dummies, 0.25)
        idea = pd.DataFrame([ideaTags, ideaTagOutlier]).T
        idea.columns=['tags', 'idea_outlier_measure']

        dfnerd = pd.DataFrame()
        if query_nerd:
            nerdC = nerd.NerdC()
            dfnerd = cluster.ips.apply(nerdC.ip_req)
            clusters = clusters.assign(ip_blocks=clusters['size'] / (clusters['size'] * dfnerd.apply(lambda x: len(set(x['ipblock'])))))

        clusters = clusters.join(idea).join(grey)

        return clusters, series, ipseries, dfnerd


def run(argv):
    # argv = ['./',  # path
    #         data_loc,  # data loc
    #         analysis_date[0],  # dfrom_to
    #         analysis_date[1],  # dto
    #         5 / 96,  # min act
    #         91 / 96,  # max act
    #         0.05,  # dist_thresh
    #         'False',  # dowload 3.rd party data / extended analysis
    #         'hdbscan']  # method

    time_window = argv[2].split("_")

    for day in pd.date_range(time_window[0], time_window[-1]):
        print("Processing files")
        (df, file_list) = load_files(argv[1], day.date().isoformat(), day.date().isoformat())

        print("Clustering")
        tc = TemporalClusterer(min_activity=argv[4], max_activity=argv[5], dist_threshold=argv[6],
                               method=argv[8], batch_limit=40000, prune_distmat=False)
        # prune=argv[8]=='True')
        df['labels'] = tc.fit_transform(df, [])
        print("Running post process")
        (clusters, series, ipseries, dfnerd) = tc.post_process3p(df, file_list,
                                                                 query_nerd=(argv[7] == 'True'),
                                                                 third_party_path=f'{argv[1]}/3rdParty/')

        # Ranking of clusters, to pick what to focus on
        #top10 = clusters.sort_values(by=['score', 'size'], ascending=False).head(10)

        intervals = sample_intervals(series, file_list[0], tc.aggregation)
        # tc.aggregations should be same as with series

        # only if you want flows and more data
        df_flows = pd.Series(dtype=object)
        dfnerd = pd.DataFrame()

        # Works well for one day clustering # needs rework
        if argv[7] == 'True' and False:
            df_flows = clusters_get_flows(top10['ips'], intervals.loc[top10.index])

            df_ip = df.loc[df['labels'] > -1, ['ip', 'labels']].loc[df['labels'] > -1]
            df_ip = df_ip.groupby('ip').agg(cluster=('labels', min))
            nerdC = nerd.NerdC()
            dfnerd = nerdC.ip_req(df_ip.index.values)

            dfnerd['cluster'] = df_ip
            del df_ip

            df_flow_views = flows_get_views(df_flows)

            store_named_df(f'./{argv[1]}/{argv[3]}/{file_list[0]}_{file_list[-1]}/',
                           dict(zip(['flows', 'viewsrcip', 'viewsrcport'],
                                    df_flows, df_flow_views[0], df_flow_views[1])))

        store_named_df(f'./{argv[1]}/{argv[3]}/{file_list[0]}_{file_list[-1]}/',
                       dict(zip(['events', 'clusters', 'series', 'dfnerd'],
                                [df.loc[df['labels'] > -1, :], clusters, series, dfnerd])))
        plt.figure()
        sns.heatmap(series)
        #print(clusters.loc[top10.index, ['size', 'events', 'ips', 'idea_tags']])

    return


if __name__ == '__main__':

    from IPython import get_ipython

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'qt')

    run(sys.argv)

    # %%
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

