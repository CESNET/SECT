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

# %%


def new_ips_hourly(ips_in_hour):
    # ips_in_hour = df[['ip', 'hour']].groupby('hour').agg([set, lambda x: len(set(x))])

    tmp = ips_in_hour.iloc[0, 0]
    lst = []
    for a in ips_in_hour.iloc[:, 0]:
        lst.append((len(set.difference(a, tmp))))
        tmp = tmp.union(a)

    sns.lineplot(data=np.array(lst))
    plt.title('Hourly new unique ip\'s')
    return


#%%
sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

days = 7

file_list = ['2019-09-{:02d}'.format(x) for x in range(31, 31)]+['2019-10-{:02d}'.format(x) for x in range(8, 15)]

df = pd.DataFrame()
for file_name in file_list:
    df = pd.concat([df, pd.read_pickle('./data/' + file_name + '.pcl')], ignore_index=True)
# %%

tfrom = datetime.datetime.fromisoformat('{} 00:00:00'.format(file_list[0])).timestamp()
tto = datetime.datetime.fromisoformat('{} 23:59:59'.format(file_list[-1])).timestamp()

df = df.loc[(df['timestamp'] >= tfrom) & (df['timestamp'] < tto), :]
df['hour'] = ((df.timestamp - tfrom) / 3600).astype(np.int)


# %%
df['day'] = (df.hour / 24).astype(np.int)
lst = df[['ip', 'day']].groupby('ip').agg(lambda x: (list(set(x))))

preclust = pd.DataFrame(data=np.stack(lst.day.apply(get_bin_series, args=[days])), index=lst.index)
preclust['ip'] = lst.index
ip_fractions = preclust.groupby(list(range(0, days)))['ip'].apply(list)

aggr=900
c = tc.TemporalClusterer(min_cluster_size=5, min_events=5, dist_threshold=0.005, aggregation=aggr, max_activity=0.5)

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
#%%

clusters = (df.loc[df.labels > -1]
            .groupby('labels')
            .agg({'ip': [set, lambda x: len(set(x)), 'count'], 'timestamp': [min, max], 'origin': set, 'type': set})
            #.rename({'ip': ('ip', 'ip_count'), 'timestamp': ('min', 'max'), 'origin': 'sources', 'type': 'evt_types'})
            )

clusters.sort_values(('ip', '<lambda>'), ascending=False, inplace=True)


series = df.loc[df.labels > -1]\
    .groupby(['labels', 'ip'])['timestamp']\
    .agg(list)\
    .apply(lambda x: np.array(get_bin_series(x, c.vect_len), dtype=np.float))\
    .groupby('labels')\
    .agg(list)\
    .apply(lambda x: np.sum(x, axis=0)/len(x))

fingerprint = pd.DataFrame(np.stack(series), index=series.index)

#ordering = pd.DataFrame(data=np.stack(fingerprint.apply(lambda x: np.correlate(x, x, mode='full')[len(x):len(x)+16], axis=1)), index=fingerprint.index)

ordering = pd.DataFrame(data=np\
        .stack(fingerprint\
        .apply(inter_arrival, args=[0.0], axis=1)),
    #lambda x: np.correlate(x, x, mode='full')[len(x):len(x)+16],
    index=fingerprint.index)

#ordering = ordering.apply(lambda x: x/max(x,axis=1))
ordering.sort_values(by=list(ordering.columns), ascending=False, inplace=True)

#%%
#%matplotlib qt


plt.figure()
sns.heatmap(data=fingerprint.iloc[ordering.index, :])
plt.figure()
sns.heatmap(data=fingerprint)



    #get_beginning(fingerprint.iloc[2,:],datetime.datetime.fromisoformat(file_list[0]).timestamp(),aggr,1)
