import numpy as np
import pandas as pd
import datetime
from preprocess import *

import matplotlib.pyplot as plt
import seaborn as sns

def filter_frame(df, tfrom, tto, agg_secs=900, min_count=5):
    df = pd.read_csv('../data/2019-08-01_test2.csv')
    df = df[['ip', 'timestamp']]

    # A = get_aggregated(df, '2019-08-01 00:00:00','2019-08-02 00:00:00')
    # A = get_aggregated(df)

    tfrom = datetime.datetime.fromisoformat('2019-08-01 00:00:00').timestamp()
    tto = datetime.datetime.fromisoformat('2019-08-02 00:00:00').timestamp()

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


def dc_liberouter_filter_string(val):
    return 'ip in {}'.format(list(val)).replace('\'', '')


def inter_arrival(x, thr):
    idxes=np.nonzero(np.diff(x) > thr)
    idxes=idxes-idxes[0][0] #start from fist event
    inter=np.diff(idxes)
    return [np.std(inter), np.mean(inter)]


#Does not merge intervals!. Second column in result marks end of 15 min interval
#todo maybe merge intervals
def get_beginning(x, first, agg, offset):
    x=pd.Series(data=np.concatenate(([0], x)), index=range(0-agg,(len(x))*agg,agg))
    thr=x[x > 0].median()

    blocks = x[x > thr]
    if len(blocks) < 1:
        blocks = x[x >= thr]

    gaps = pd.Series(data=np.diff(blocks.index), index = blocks.index[:-1])

    take = blocks[(gaps>offset*agg).index].index

    block_e = [datetime.datetime.fromtimestamp(val).isoformat()\
               for val in first+np.array(take, dtype=np.int)+agg]

    block_s = [datetime.datetime.fromtimestamp(val).isoformat()\
               for val in first+(np.array(take, dtype=np.int)-offset*agg)]

    lst = np.array([block_s, block_e]).T

    save = [0]
    edit = 0
    if len(lst) > 0:
        for t in range(1,len(lst)):
            if lst[edit][1] >= lst[t][0]:
                lst[edit][1] = lst[t][1]
            else:
                edit = t
                save.append(edit)

    return lst[save]


def plot_clusters(clusters, fingerprint, aggr, file_list, clust_list):

    plt.figure()
    sns.heatmap(fingerprint.loc[clust_list,:])
    plt.xticks(list(range(0, 671, 12)),
               [datetime.datetime.fromtimestamp(
                   datetime.datetime.fromisoformat(
                       file_list[0]).timestamp() + x * aggr).\
               strftime("%b %d %H:%M") for x in range(0, 671, 12)])

