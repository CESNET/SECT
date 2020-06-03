import numpy as np
import pandas as pd
import datetime
from utils import *

from preprocess import *
import nerd

from scipy.spatial.distance import squareform, pdist
import sklearn.cluster as sc

from sklearn import metrics

from scipy.stats import binom
from scipy.special import comb

# import umap
import hdbscan
import dCollector
import graphing



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
        idx = 0

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
                    #ws_div *= 2 ** x

                    idx = 0

                    for y in range(0, len(series.columns), 2 ** x):
                        ws_div.loc[:] += ((work_set.iloc[:, y:y + 2 ** x].apply(max, axis=1) > 0).astype(np.int) *
                                         (2 ** idx))
                        idx += 1

                    # Update new division
                    division.loc[ws_div.index] = ws_div
                    stopper = False

            if stopper:
                break

        return division

    def transform(self, df, tfrom, tto):
        """
        Aggregates event table to activity vectors by IP and produces features and
        extra informatrion related to them.
        Produces 4 output dataframes for vectors, features, type and origin
        """

        dfc = df.loc[(df.timestamp >= tfrom.timestamp()) & (df.timestamp < tto.timestamp()), :]

        dfc = dfc.assign(type=df['type'].apply(lambda x: x.replace('[', '').replace(']', '').replace(' ', '')
                                              .replace("'", '').replace('"', '').replace(', ', '-')),
                         origin=df['origin'].apply(lambda x: x.split(',')[-1].rstrip(']').rstrip('}')
                                                  .split(':')[-1].strip(' ').replace('\'', ''))
                        )

        #type and origin counts per ip
        type = pd.get_dummies(dfc['type'])
        type = type.groupby(dfc.ip).agg(sum)

        origin = pd.get_dummies(dfc['origin'])
        origin = origin.groupby(dfc.ip).agg(sum)

        dfc['slot'] = np.floor((df.timestamp - tfrom.timestamp()) / self.aggregation).astype(np.int)

        feature = dfc.slot.groupby(dfc['ip']).agg([list, 'count'])

        # expected vector length
        self.vect_len = np.int((tto.timestamp()-tfrom.timestamp())/self.aggregation)

        if self.min_events is None:
            self.min_events = np.ceil(self.vect_len * self.min_activity)
        else:
            self.min_events = np.int(self.min_events)

        feature['series'] = feature.list.apply(get_bin_series, args=[self.vect_len])
        feature['activity'] = feature.series.apply(np.sum)
        feature['blocks'] = feature.series.apply(count_blocks)

        act_vect = pd.DataFrame(data=np.stack(feature['series']), dtype=np.bool, index=feature.index)

        del feature['series']
        del feature['list']
        del dfc['slot']

        return act_vect, feature, type, origin


    def fit_transform(self, act_vect, feature):
        """
        Cluster activity vectors by selected method. This function translates method
        settings to each an every type of clustering used, and makes it feasible to cluster
        large dataset, on smaller machine.
        Before clustering divide space to weakly dependant sets and cluster per
        partes, this way huge memory requirements can be restricted to maximum
        given by limit.
        """
        vect_all = act_vect # store the index to all rows
        act_vect = act_vect.loc[(feature['activity'] >= (self.min_activity * self.vect_len)),:]\
                            .loc[(feature['activity'] < np.ceil(self.max_activity * self.vect_len)),:]

        labels = pd.Series(data=-1, index=act_vect.index)

        # limit  #40000 - 12.8 GiB pairwise matrix size

        if len(act_vect) > 0:

            division = pd.Series(data=0, index=act_vect.index)

            if len(act_vect) > self.limit:
                # hash space, get smaller chunks of data to work on per partes
                division = self.series_divide(division, act_vect, limit=self.limit * 0.7)

            z = division.value_counts()

            labels_ofs = 0

            for group in z.index:

                vect = act_vect.loc[division == group]

                #pairwise = pd.DataFrame()
                # stack of features with ip in time index
                if len(vect) == 0:
                    continue
                # if group is too big, take head, most active with most blocks or sample...
                if len(vect) > self.limit:
                    # data.sort_values(inplace=True, ascending=False, by=['activity', 'blocks'])
                    vsort = feature.loc[vect.index,:].sort_values(ascending=False, by=['activity','count'])

                    vect = vect.loc[vsort.head(self.limit).index, :]
                    print(f"Data too big, reduced count from {len(vsort)} to {self.limit}", file=sys.stderr)

                if len(vect) > 1:
                    pairwise = pd.DataFrame(data=squareform(pdist(vect, self.metric)),
                                            index=vect.index,
                                            columns=vect.index,
                                            dtype=np.double)

                    # calculate distance matrix and prepare for clustering
                    # prune features/distance matrix
                    if self.dist_threshold is not None and self.prune:
                        matches = pairwise.apply(lambda x: np.sum(x <= self.dist_threshold)) \
                            .where(lambda x: x >= self.min_cluster_size) \
                            .dropna()

                        pairwise = pairwise.loc[matches.index, matches.index]

                    if len(pairwise) > 1:
                        # try agglomerative
                        if self.method == 'hdbscan':
                            labels_part = hdbscan.HDBSCAN(
                                min_cluster_size=self.min_cluster_size,
                                metric='precomputed',
                                # cluster_selection_method='leaf'
                            ).fit_predict(pairwise)  # why ?

                            labels_part = pd.Series(labels_part + labels_ofs, index=pairwise.index)


                        elif self.method == 'dbscan':
                            eps = self.dist_threshold
                            if self.dist_threshold is None:
                                eps = self.default_dist_threshold
                            labels_part = sc.DBSCAN(
                                eps=eps, min_samples=self.min_cluster_size,
                                metric='precomputed'
                            ).fit_predict(pairwise)

                            labels_part = pd.Series(labels_part + labels_ofs, index=pairwise.index)

                        elif self.method == 'optics':
                            eps = self.dist_threshold
                            if self.dist_threshold is None:
                                eps = self.default_dist_threshold
                            labels_part = sc.OPTICS(
                                max_eps=eps,  # min_samples=self.min_cluster_size,
                                min_cluster_size=self.min_cluster_size,
                                metric='precomputed'
                            ).fit_predict(pairwise)

                            labels_part = pd.Series(labels_part + labels_ofs, index=pairwise.index)

                        elif self.method == 'agglomerative':
                            AC = sc.AgglomerativeClustering(
                                affinity='precomputed',
                                linkage='complete',
                                distance_threshold=self.dist_threshold,
                                n_clusters=None
                            )
                            labels_part = AC.fit_predict(pairwise)

                            labels_part = pd.Series(labels_part + labels_ofs, index=pairwise.index)
                            # Filter smaller clusters then minimum
                            labels_part = labels_part.loc[labels_part.map(labels_part.value_counts()) >= self.min_cluster_size]


                        elif self.method == 'match':
                            AC = sc.AgglomerativeClustering(
                                affinity='precomputed',
                                linkage='complete',
                                distance_threshold=0.001,
                                n_clusters=None
                            )
                            labels_part = AC.fit_predict(pairwise)  # why ?

                            labels_part = pd.Series(labels_part + labels_ofs, index=pairwise.index)
                            # Filter smaller clusters then minimum
                            labels_part = labels_part.loc[labels_part.map(labels_part.value_counts()) >= self.min_cluster_size]


                        labels_ofs = labels_part.max() + 1
                        labels.loc[labels_part.index] = labels_part

        return labels

    def postprocess(self, labels, act_vect, feature, type, origin, tfrom, tto, query_nerd=False, third_party_path='./data/3rdParty'):
        """ Taking results from fit_transform, this function generates feature vectors for clusters
            attaches the information from 3rd party and summarizes what is know about clusters into few tables
        """
        
        plabels = labels.loc[labels>-1]
        pvect = act_vect.loc[plabels.index,:]

        feature['ip'] = feature.index

        clusters = feature.groupby(plabels).agg(
            ips = ('ip', lambda x: list(x)),
            size = ('ip', lambda x: len(list(x))),
            events = ('count', np.mean),
            activity = ('activity', np.mean),
            blocks = ('blocks', np.mean),
        )
        clusters['silhouette'] =  pd.Series(metrics.silhouette_samples(pvect, plabels, metric=self.metric), index=plabels.index)\
            .groupby(plabels).agg(np.mean)


        series = act_vect

        act_vect.columns = pd.DatetimeIndex(
            pd.date_range(start=tfrom,
                          end=tto,
                          periods=self.vect_len + 1)[:-1])\
            .strftime('%Y-%m-%d %H:%M')

        series = (act_vect.groupby(plabels).agg(sum).T/plabels.value_counts()).T


        # generate stats within clusters
        # cohesion lower better
        ipsAll = act_vect.index.to_series()
        stats = pd.DataFrame()

        stats['activity_dev'] = pvect.groupby(plabels).agg(np.std).T.sum()
        stats['idea_dev'] = type.groupby(plabels).agg(np.std).T.sum()
        stats['detector_dev'] = origin.groupby(plabels).agg(np.std).T.sum()

        ideaTags, ideaTagOutlier = process_tags(ipsAll, clusters, type, 0.05)
        idea = pd.DataFrame([ideaTags, ideaTagOutlier]).T
        idea.columns = ['idea_tags', 'idea_outlier_measure']
        idea['idea_outlier_measure'] = idea['idea_outlier_measure'].astype(np.float)

        originTags, originTagOutlier = process_tags(ipsAll, clusters, origin, 0.05)
        detector = pd.DataFrame([originTags, originTagOutlier]).T
        detector.columns = ['detector_tags', 'detector_outlier_measure']
        detector['detector_outlier_measure'] = detector['detector_outlier_measure'].astype(np.float)

        stats = stats.join(idea).join(detector)

        if (third_party_path):
            trd = load_named_df(third_party_path, ['greynoise_ip_db_indication',
                                                   #'mentat_ip_db_indication',
                                                   'nerd_ip_db_indication'])

            trd['greynoise_ip_db_indication'].index=trd['greynoise_ip_db_indication'].index.to_series().transform(str)
            trd['nerd_ip_db_indication'].index=trd['nerd_ip_db_indication'].index.to_series().transform(str)

            stats['grey_dev'] = trd['greynoise_ip_db_indication'].groupby(plabels).agg(np.std).T.sum()
            stats['nerd_dev'] = trd['nerd_ip_db_indication'].groupby(plabels).agg(np.std).T.sum()

            greyTags, greyTagOutlier = process_tags(ipsAll, clusters, trd['greynoise_ip_db_indication'], 0.05)
            grey = pd.DataFrame([greyTags, greyTagOutlier]).T
            grey.columns = ['grey_tags', 'grey_outlier_measure']
            grey['grey_outlier_measure'] =  grey['grey_outlier_measure'].astype(np.float)

            #mentatTags, mentatTagOutlier = process_tags(dfAll, clusters, trd['mentat_ip_db_indication'], 0.25)
            #mentat = pd.DataFrame([mentatTags, mentatTagOutlier]).T
            #mentat.columns = ['mentat_tags', 'mentat_outlier_measure']

            nerdTags, nerdTagOutlier = process_tags(ipsAll, clusters, trd['nerd_ip_db_indication'], 0.05)
            nerd3p = pd.DataFrame([nerdTags, nerdTagOutlier]).T
            nerd3p.columns = ['nerd_tags', 'nerd_outlier_measure']
            nerd3p['nerd_outlier_measure'] =  nerd3p['nerd_outlier_measure'].astype(np.float)

            stats = stats.join(grey).join(nerd3p)  # .join(mentat)

        dfnerd = pd.DataFrame()
        if query_nerd:
            nerdC = nerd.NerdC()
            dfnerd = clusters.ips.apply(nerdC.ip_req)
            clusters = clusters.assign(
                ip_blocks=clusters['size'] / (clusters['size'] * dfnerd.apply(lambda x: len(set(x['ipblock'])))))


        return clusters, series, stats, dfnerd



    def probEstimate(self, labels, clusters, feat):
        trials = feat.index.to_series().groupby(feat.activity).agg('count')
        p = pd.Series(data=1 / comb(int(self.vect_len), range(1, int(self.vect_len))),
                      index=range(1, self.vect_len))
        # -1 to start, because weird counting
        return clusters.apply(lambda x: binom._pmf(int(x['size'])-1, trials[int(x.activity)], p[int(x.activity)]), axis=1)

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
                               method=argv[8], batch_limit=20000, prune_distmat=True, min_cluster_size=2)
        # prune=argv[8]=='True')
        vect, feat, type, origin = tc.transform(df, day, day + datetime.timedelta(days=1)) # not needed to redo if aggregation does not change

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



if __name__ == '__main__':

    from IPython import get_ipython

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'qt')

    #run(sys.argv)
    argv=sys.argv

    run(argv)
    # time_window = argv[2].split("_")
    #
    # for day in pd.date_range(time_window[0], time_window[-1]):
    #     print("Processing files")
    #     (df, file_list) = load_files(argv[1], day.date().isoformat(), day.date().isoformat())
    #
    #     print("Clustering")
    #     tc = TemporalClusterer(min_activity=argv[4], max_activity=argv[5], dist_threshold=argv[6],
    #                            method=argv[8], batch_limit=20000, prune_distmat=True, min_cluster_size=2)
    #     # prune=argv[8]=='True')
    #     vect, feat, type, origin = tc.transform(df, day, day + datetime.timedelta(days=1)) # not needed to redo if aggregation does not change
    #
    #     labels = tc.fit_transform(vect, feat)
    #
    #     print("Running post process")
    #     (clusters, series, stats, dfnerd) = tc.postprocess(labels,
    #                                                      vect, feat, type, origin,
    #                                                      day, day + datetime.timedelta(days=1),
    #                                                      query_nerd=False,
    #                                                      third_party_path=f'{argv[1]}/3rdParty')
    #
    #
    #     store_named_df(f'./{argv[1]}/{argv[3]}/{file_list[0]}_{file_list[-1]}/',
    #                    dict(zip(['clusters', 'series', 'stats', 'ipseries', 'feature', 'idea', 'detector'],
    #                             [clusters, series, stats, vect, feat, type, origin])))
    #     if len(dfnerd) > 0:
    #         store_named_df(f'./{argv[1]}/{argv[3]}/{file_list[0]}_{file_list[-1]}/',
    #                        dict(zip(['dfnerd'], [dfnerd])))


