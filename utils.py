import numpy as np
import pandas as pd
import datetime
from pathlib import Path

import preprocess
import dCollector
import nerd

import matplotlib.pyplot as plt
import seaborn as sns

timeFormat = "%Y-%m-%d %H:%M:%S"
dateFormat = "%Y-%m-%d"


def filter_frame(df, tfrom, tto, agg_secs=900, min_count=5):
    df = pd.read_csv('../data/2019-08-01_test2.csv')
    df = df[['ip', 'timestamp']]

    # A = get_aggregated(df, '2019-08-01 00:00:00','2019-08-02 00:00:00')
    # A = get_aggregated(df)

    tfrom = datetime.datetime.strptime('2019-08-01 00:00:00', timeFormat).timestamp()
    tto = datetime.datetime.strptime('2019-08-02 00:00:00', timeFormat).timestamp()

    agg_secs = 900
    days_proc = 1
    corr_window = 1.25

    A = df.copy()
    A = df[df['timestamp'] >= tfrom]
    A = A[A['timestamp'] < tto]

    A['timestamp'] = A.timestamp - tfrom
    A['timestamp'] = np.floor(A.timestamp / agg_secs)
    Ag = A.groupby('ip')['timestamp'].agg([list, 'count'])

    # df['agg']=np.floor(df['timestamp']/900)
    # exp=df.groupby('agg')['ip'].agg([set])
    # sns.barplot(A['timestamp','count'])

    data = Ag[Ag['count'] > 10]  # prefiltered data

    data.head()
    print(data['count'].describe())
    data['count'].hist(bins=list(range(0, 75, 5)))
    # plt.figure()
    evt_signal = A.groupby('timestamp')['ip'].agg('count')
    # sns.lineplot(data=evt_signal)

    data['series'] = data['list'].apply(preprocess.get_bin_series, args=[np.int((days_proc * 24 * 3600) / agg_secs)])
    data.sort_values('count', inplace=True, ascending=True)
    vect = pd.concat([pd.DataFrame(index=data.index, data=np.stack(data.series)), data['count']], axis=1)


def inter_arrival(x, thr):
    idxes = np.nonzero(np.diff(x) > thr)
    idxes = np.subtract(idxes, idxes[0][0])  # start from fist event
    inter = np.diff(idxes)
    return [np.std(inter), np.mean(inter)]


def sample_first_last_mid(df):
    if len(df) > 2:
        return df.iloc[[0, int(len(df)/2), len(df)-1], :]
    else:
        return df


def sample_intervals(series, first, aggregation=900, pre_block_pad=1, sample_size=3):
    first_timestamp = datetime.datetime.strptime(first, dateFormat).timestamp()
    # intervals=df.apply(lambda x: get_beginning(x, first, aggregation, pre_block_pad).sample(3))
    intervals = series.apply(
        lambda x: sample_first_last_mid(
            pd.DataFrame(
                data=get_intervals(x, first=first_timestamp, agg=aggregation, offset=pre_block_pad),
                columns=('from', 'to')
            ).applymap(lambda y: pd.Timestamp(y).timestamp()).astype(int)
        ),
        axis=1)

    return pd.Series(intervals, index=series.index)


def get_intervals(x, first, agg, offset):
    x = pd.Series(data=np.concatenate(([0], x)), index=range(0 - agg, (len(x)) * agg, agg))
    thr = x[x > 0].median()

    blocks = x[x > thr]
    if len(blocks) < 1:
        blocks = x[x >= thr]

    gaps = pd.Series(data=np.diff(blocks.index), index=blocks.index[:-1])

    take = blocks[(gaps > offset * agg).index].index

    block_e = [datetime.datetime.fromtimestamp(val).isoformat() \
               for val in first + np.array(take, dtype=np.int) + agg]

    block_s = [datetime.datetime.fromtimestamp(val).isoformat() \
               for val in first + (np.array(take, dtype=np.int) - offset * agg)]

    lst = np.array([block_s, block_e]).T

    save = [0]
    edit = 0
    if len(lst) > 0:
        for t in range(1, len(lst)):
            if lst[edit][1] >= lst[t][0]:
                lst[edit][1] = lst[t][1]
            else:
                edit = t
                save.append(edit)

    return lst[save]


def plot_clusters(clusters, fingerprint, aggr, file_list, clust_list):
    plt.figure()
    sns.heatmap(fingerprint.loc[clust_list, :])
    plt.xticks(list(range(0, 671, 12)),
                [datetime.datetime.fromtimestamp(
                datetime.datetime.strptime(file_list[0], timeFormat).timestamp() + x * aggr).\
                    strftime("%b %d %H:%M") for x in range(0, 671, 12)])


def cumulative_unique_counts(set_list):
    # ips_in_hour = df[['ip', 'hour']].groupby('hour').agg([set, lambda x: len(set(x))])

    tmp = set_list.iloc[0, 0]
    lst = []
    for a in set_list.iloc[:, 0]:
        lst.append((len(set.difference(a, tmp))))
        tmp = tmp.union(a)

    sns.lineplot(data=np.array(lst))
    plt.title('Hourly new unique ip\'s')
    return


def load_files(working_dir, date_from, date_to, idea_dir=None):

    if idea_dir == None:
        idea_dir = working_dir+'\\IDEA'

    file_list = [x.date().isoformat() for x in pd.date_range(date_from, date_to)]
    days = len(file_list)

    df = pd.DataFrame()

    for file_name in file_list:

        file_obj = Path(working_dir + '\\' + file_name + '.pcl')

        if file_obj.is_file():
            df = pd.concat([df, pd.read_pickle(file_obj)], ignore_index=True)

        elif not file_obj.exists():
            #we need to preprocess the idea file
            idea_file_obj = Path(idea_dir + "\\" + file_name + '.idea')
            gz_file_obj = Path(idea_dir + "\\" + file_name + '.gz')

            df_prep = pd.DataFrame()

            if idea_file_obj.is_file():
                df_prep = preprocess.preprocess(str(idea_file_obj))
            elif gz_file_obj.is_file():
                df_prep = preprocess.preprocess(str(gz_file_obj))
            else:
                print(f"Can't process idea file {str(idea_file_obj).rsplit('.')[1]}.(pcl|gz)")

            df_prep.to_pickle(file_obj)

            df = pd.concat([df, df_prep], ignore_index=True)
        else:
            print(f"Can't process file {str(file_obj).rsplit('.')[0]}.(pcl|gz), it is not a file")

    tfrom = datetime.datetime.strptime('{} 00:00:00'.format(file_list[0]),timeFormat).timestamp()
    tto = datetime.datetime.strptime('{} 23:59:59'.format(file_list[-1]), timeFormat).timestamp()

    return df.loc[(df['timestamp'] >= tfrom) & (df['timestamp'] < tto), :], file_list


def rank_clusters(cluster, series, cluster_type_count, cluster_origin_count):

    score = pd.DataFrame(index=cluster.index)
    score['size_in_quantile_95'] = (cluster['size'] >= cluster['size'].quantile(.95))
    score['events_in_quantlie_95'] = (cluster['events'] >= cluster['events'].quantile(.95))
    inter_arrival_dev = series.apply(lambda x: inter_arrival(x, 0.3)[1], axis=1)
    iaq = inter_arrival_dev.quantile(.80)
    score['series_is_irregular'] = (inter_arrival_dev > iaq)
    ctq = cluster_type_count.quantile(.95)
    score['type_count_in_quantile_95'] = cluster_type_count.apply(lambda x: np.sum(x > ctq), axis=1)
    otq = cluster_origin_count.quantile(.95)
    score['detector_count_in_quantile_95'] = cluster_origin_count.apply(lambda x: np.sum(x > otq), axis=1)

    #score['type_tag'] = False
    score['series_is_not_consistent'] = series.apply(lambda x: x.loc[x > 0].mean() < 0.7, axis=1) * -1

    if True:
        nerdC = nerd.NerdC()
        df_nerd = cluster.ips.apply(nerdC.ip_req)
        score['ipblocks_ip_count_ratio'] = cluster['size']/(cluster['size']*df_nerd.apply(lambda x: len(set(x['ipblock'])))) > 0.7

    score_sum = score.apply(np.sum, axis=1)
    score_sum.sort_values(inplace=True, ascending=False)

    tags = pd.DataFrame(index=cluster.index)
    for x in score.columns:
        tags[x] = score[x].apply(lambda y: [x] if y > 0 else [])

    for x in cluster_type_count.columns:
        tags[x] = cluster_type_count[x].apply(lambda y: [x] if y > ctq[x] else [])

    for x in cluster_origin_count.columns:
        tags[x] = cluster_origin_count[x].apply(lambda y: [x] if y > otq[x] else [])

    tag_list = tags.apply(lambda row: [item for sublist in row for item in sublist], axis=1)

    return score_sum, score, tag_list


def clusters_get_flows(cluster, interval, dc_conn=None):

    if dc_conn is None:
        dc_conn = dCollector.dc()

    flow_sample = pd.Series(index=interval.index, dtype=object)

    #iterate not apply for better readability
    for x in interval.index:
        flow_sample[x] = pd.DataFrame()
        for val in interval[x].index:
            filter_str = dCollector.dc_liberouter_filter_string(cluster.loc[x])
            df = dc_conn.filterFlows('src ' + filter_str,
                                     interval[x].loc[val, 'from'],
                                     interval[x].loc[val, 'to'])
            flow_sample[x] = pd.concat([flow_sample[x], df])
            df = dc_conn.filterFlows('dst ' + filter_str,
                                     interval[x].loc[val, 'from'],
                                     interval[x].loc[val, 'to'])
            flow_sample[x] = pd.concat([flow_sample[x], df])

    return flow_sample


def bitwise_or_series(x):
    res = 0
    for val in x:
        res = np.bitwise_or(res, val)
    return res


def to_str_flags(x):
    res = ''
    lst = 'CEUAPRSF'
    val = np.binary_repr(x, width=8)
    for x in range(0, 8):
        if val[x] == '1':
            res += lst[x]
        else:
            res += '.'
    return res


def flows_aggregate(flows, by=['srcip'], target='dstport', head_n=5):

    #flows=('packets', 'count')
    #duration=('duration', 'sum')
    #packets=('packets', 'sum')
    #bytes=('bytes', 'sum')
    #flags=('flags', np.bitwise_or)
    #dstip_count = ('dstip', lambda x: len(set(x))),
    #dstip_top = ('dstip', lambda x: (pd.value_counts(x).head(5))),
    #dstport_count = ('dstport', lambda x: len(set(x))),
    #dstport_top = ('dstport', lambda x: (pd.value_counts(x).head(5))),

    other_side = 'dstip'
    if by.__contains__('dstip'):
        other_side = 'srcip'

    dfa = flows.groupby(by).agg(
            flows=('packets', 'count'),
            ip_count=(other_side, lambda x:len(set(x))),
            ip_top=(other_side, lambda x: str(pd.value_counts(x).head(head_n))),
            target_count=(target, lambda x: len(set(x))),
            target_top=(target, lambda x: str(pd.value_counts(x).head(head_n))),

            duration=('duration', 'sum'),
            packets=('packets', 'sum'),
            bytes=('bytes', 'sum'),
            flags=('flags', bitwise_or_series)
        )

    dfa['duration'] = dfa['duration'].apply(lambda x: datetime.timedelta(milliseconds=x))
    dfa['flags'] = dfa['flags'].apply(to_str_flags)
    dfa['bps'] = dfa.bytes/(dfa.duration.apply(datetime.timedelta.total_seconds))
    dfa['pps'] = dfa.packets/(dfa.duration.apply(datetime.timedelta.total_seconds))
    dfa['bpp'] = dfa.bytes/dfa.packets

    dfa.sort_values(inplace=True, ascending=False, by='flows')

    return dfa


def flows_get_views(flows):

    bysrcip = flows.apply(flows_aggregate)
    bysrcport = flows.apply(flows_aggregate, by=['srcport'])
    bydstport = flows.apply(flows_aggregate, by=['dstport'])

    return bysrcip, bysrcport, bydstport
    #return [flows_aggregate(flows),\
    #       flows_aggregate(flows, by=['srcport'])]
    #       #flows_aggregate(flows, by=['src'], target='srcip')


def store_analysis(path, df, clusters, series, dfnerd, dfflows, dfviewsrcip, dfviewsrcport):

    where = Path(path)
    where.mkdir(parents=True, exist_ok=True)

    df.to_pickle(path+'/events.pcl')
    clusters.to_pickle(path+'/clusters.pcl')
    series.to_pickle(path+'/series.pcl')
    dfnerd.to_pickle(path+'/nerd.pcl')
    dfflows.to_pickle(path+'/flows.pcl')
    dfviewsrcip.to_pickle(path+'/viewsrcip.pcl')
    dfviewsrcport.to_pickle(path+'/viewsrcport.pcl')

    return


def load_analysis(path):
    where = Path(path)
    if where.is_dir():
        df = pd.read_pickle(path+'/events.pcl')
        clusters = pd.read_pickle(path+'/clusters.pcl')
        series = pd.read_pickle(path+'/series.pcl')
        dfnerd = pd.read_pickle(path+'/nerd.pcl')
        flows = pd.read_pickle(path+'/flows.pcl')
        dfviewsrcip= pd.read_pickle(path+'/viewsrcip.pcl')
        dfviewsrcport= pd.read_pickle(path+'/viewsrcport.pcl')

        return df, clusters, series, dfnerd, flows, dfviewsrcip, dfviewsrcport
    else:
        print('Analysis is not done at that time range')
        return None