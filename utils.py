import numpy as np
import pandas as pd
import datetime
from pathlib import Path
from preprocess import *

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

    data['series'] = data['list'].apply(get_bin_series, args=[np.int((days_proc * 24 * 3600) / agg_secs)])
    data.sort_values('count', inplace=True, ascending=True)
    vect = pd.concat([pd.DataFrame(index=data.index, data=np.stack(data.series)), data['count']], axis=1)


def inter_arrival(x, thr):
    idxes = np.nonzero(np.diff(x) > thr)
    idxes = np.subtract(idxes, idxes[0][0])  # start from fist event
    inter = np.diff(idxes)
    return [np.std(inter), np.mean(inter)]


def sample_first_last_mid(df):
    if len(df) > 3:
        return df.iloc[[0, int(len(df)/2), -1], :]
    else:
        return df


def sample_intervals(df, first, aggregation=900, pre_block_pad=1, sample_size=3):
    first_timestamp = datetime.datetime.strptime(first, dateFormat).timestamp()
    # intervals=df.apply(lambda x: get_beginning(x, first, aggregation, pre_block_pad).sample(3))
    intervals = df.apply(
        lambda x: (sample_first_last_mid(pd.DataFrame(get_intervals(x, first=first_timestamp, agg=aggregation, offset=pre_block_pad)))
                   .applymap(lambda y: pd.Timestamp(y).timestamp()).astype(int)).values
        , axis=1)

    return np.array(intervals)


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
            #we need to preprocess the idea filee
            idea_file_obj = Path(idea_dir + "\\" + file_name + '.idea')
            gz_file_obj = Path(idea_dir + "\\" + file_name + '.gz')

            df_prep = pd.DataFrame()

            if idea_file_obj.is_file():
                df_prep = preprocess(str(idea_file_obj))
            elif gz_file_obj.is_file():
                df_prep = preprocess(str(gz_file_obj))
            else:
                print(f"Can't process idea file {str(idea_file_obj).rsplit('.')[1]}.(pcl|gz)")

            df_prep.to_pickle(file_obj)

            df = pd.concat([df, df_prep], ignore_index=True)
        else:
            print(f"Can't process file {str(file_obj).rsplit('.')[0]}.(pcl|gz), it is not a file")

    tfrom = datetime.datetime.strptime('{} 00:00:00'.format(file_list[0]),timeFormat).timestamp()
    tto = datetime.datetime.strptime('{} 23:59:59'.format(file_list[-1]), timeFormat).timestamp()

    return df.loc[(df['timestamp'] >= tfrom) & (df['timestamp'] < tto), :], file_list


def filter_clusters(clust, ser, c_typ, c_org):
    criterions = pd.DataFrame(index=clust.index)
    criterions['size_quantile_95'] = (clust['size'] >= clust['size'].quantile(.95))
    criterions['events_quantlie_95'] = (clust['events'] >= clust['events'].quantile(.95))
    criterions['series_irregular'] = False #TODO
    ctq = c_typ.quantile(.95)
    criterions['count_of_type_gt_than_quantile_95'] = clust.apply(lambda x: np.sum(x > ctq), axis=1)


    #TODO vyber zaujimave klastre, automaticky
    pass

def collect_flows(clusters):
    #TODO ziskaj flow vzorky pre kazdy vybrany cluster
    pass
