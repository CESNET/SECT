#%matplotlib qt

from preprocess import *
from utils import *
import TemporalClusterer as tc
import numpy as np
import pandas as pd
import datetime

import umap
import hdbscan

import matplotlib.pyplot as plt
import seaborn as sns
#from scipy.cluster.hierarchy import dendrogram, linkage

import sys
import os

#%%
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

file_list = [x.date().isoformat() for x in pd.date_range('2020-01-25', '2020-01-31')]
days = len(file_list)

df = pd.DataFrame()
for file_name in file_list:
    df = pd.concat([df, pd.read_pickle('./data/' + file_name + '.pcl')], ignore_index=True)
# %%

tfrom = datetime.datetime.fromisoformat('{} 00:00:00'.format(file_list[0])).timestamp()
tto = datetime.datetime.fromisoformat('{} 23:59:59'.format(file_list[-1])).timestamp()

df = df.loc[(df['timestamp'] >= tfrom) & (df['timestamp'] < tto), :]

IPdf = df.groupby('ip')['timestamp'].agg(lambda x: (list(x)))
IPevts = IPdf.apply(list.__len__)

plt.figure()
IPevts.hist(bins=100)

IPevts.loc[IPevts>10].hist(bins=100)

#clip out extremes... and if to analyze them, do it separately
IPdf = IPdf.loc[IPdf.apply(list.__len__) > 1]


# %%

df['day'] = (df.hour / (24*3)).astype(np.int)
lst = df[['ip', 'day']].groupby('ip').agg(lambda x: (list(set(x))))


preclust = pd.DataFrame(data=np.stack(lst.day.apply(get_bin_series, args=[np.int(np.ceil(days/3))])), index=lst.index)
preclust['ip'] = lst.index
ip_fractions = preclust.groupby(list(range(0, np.int(np.ceil(days/3)))))['ip'].apply(list)

lenvect = ip_fractions.apply(lambda x: len(x))

aggr=np.int(sys.argv[3]) #aggregate seconds
c = tc.TemporalClusterer(aggregation=aggr,
                         min_cluster_size=sys.argv[4],
                         min_events=sys.argv[5],
                         dist_threshold=sys.argv[6],
                         max_activity=sys.argv[7],
                         metric=sys.argv[8])

#%%
label_ofs = 0
df['labels'] = -1
lls = pd.Series()

for x in range(0, len(ip_fractions)):
    dfc = (df.loc[df.ip.isin(ip_fractions.iloc[x]), :]).copy() #do isin only once

    labels = c.fit_transform(dfc, []).astype(np.int)

    labels[labels >= 0] += label_ofs
    label_ofs = labels.max()+1
    lls = pd.concat([lls, labels])


df['labels']=lls

df.timestamp = np.floor((df.timestamp - df.timestamp.min()) / aggr)




clusters = (df.loc[df.labels > -1]
            .groupby('labels')
            .agg({'ip': [set, lambda x: len(set(x)), 'count'], 'timestamp': [min, max], 'origin': set, 'type': set})
            #.rename({'ip': ('ip', 'ip_count'), 'timestamp': ('min', 'max'), 'origin': 'sources', 'type': 'evt_types'})
            )

#clusters.sort_values(('ip', '<lambda>'), ascending=False, inplace=True)


series = df.loc[df.labels > -1]\
    .groupby(['labels', 'ip'])['timestamp']\
    .agg(list)\
    .apply(lambda x: np.array(get_bin_series(x, c.vect_len), dtype=np.float))\
    .groupby('labels')\
    .agg(list)\
    .apply(lambda x: np.sum(x, axis=0)/len(x))

fingerprint = pd.DataFrame(np.stack(series),
                           index=series.index,
                           columns=pd.DatetimeIndex(
                               pd.date_range(start=file_list[0],
                                             periods=c.vect_len,
                                             freq='15T')).strftime('%m/%d-%H:%M'))


#ordering = pd.DataFrame(data=np.stack(fingerprint.apply(lambda x: np.correlate(x, x, mode='full')[len(x):len(x)+16], axis=1)), index=fingerprint.index)
#
# ordering = pd.DataFrame(data=np\
#         .stack(fingerprint\
#         .apply(inter_arrival, args=[0.0], axis=1)),
#     #lambda x: np.correlate(x, x, mode='full')[len(x):len(x)+16],
#     index=fingerprint.index)
#
# #ordering = ordering.apply(lambda x: x/max(x,axis=1))
# ordering.sort_values(by=list(ordering.columns), ascending=False, inplace=True)
#
# #%%
# #%matplotlib qt
#
# plt.figure()
# sns.heatmap(data=fingerprint.iloc[ordering.index, :])
# plt.figure()
# sns.heatmap(data=fingerprint)
#
#
#
# get_beginning(fingerprint.iloc[8,:],datetime.datetime.fromisoformat(file_list[0]).timestamp(),aggr,1)

#todo make script to store results
