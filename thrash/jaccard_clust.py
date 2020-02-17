import matplotlib.pyplot as plt
#%matplotlib qt

import TemporalClusterer as tc

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from umap import UMAP

import datetime
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

import seaborn as sns
from preprocess import *
from utils import *
#%%

sns.set(style='white', context='notebook', rc={'figure.figsize': (14, 10)})

#%%

# %%
date='2019-10-24'

df = pd.read_pickle('./data/{}.pcl'.format(date))

tfrom = datetime.datetime.fromisoformat('{} 00:00:00'.format(date)).timestamp()
tto = datetime.datetime.fromisoformat('{} 23:59:55'.format(date)).timestamp()

df = df.loc[(df['timestamp'] >= tfrom) & (df['timestamp'] < tto),:] #& (df.type != 'Recon.Scanning'),:]
#df = df.loc[(df.type != 'Recon.Scanning'),:]

c = tc.TemporalClusterer(min_events=3, min_cluster_size=3, dist_threshold=0.1, max_activity=1, metric='jaccard',
                         aggregation=900)
df['labels'] = c.fit_transform(df, [])

# %%
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

#%%
#clusters['activity'] = fingerprint

#clust_18 = df.loc[(df.labels == 18)].groupby(['labels', 'ip']).agg([lambda y: sorted(set(y)), 'count'])#,lambda x: len(set(x))])
#
# sources = df.groupby('origin')['timestamp'].agg(lambda x: sorted(list(x))).apply(get_bin_series, args=[c.vect_len])
# sources = pd.DataFrame(data=np.stack(sources), index=sources.index).transpose()
# correlated = np.correlate(fingerprint.iloc[:, 0], fingerprint.iloc[:, 0], mode='full')
#
# series2 = df.loc[df.labels > -1]\
#     .groupby(['labels', 'ip'])['timestamp']\
#     .agg(list)\
#     .apply(lambda x: np.array(get_series(x, c.vect_len), dtype=np.float))\
#     .groupby('labels')\
#     .agg(list)\
#     .apply(lambda x: np.sum(x, axis=0)/len(x))
#
# fingerprint2 = pd.DataFrame(np.stack(series2), index=series2.index)

ptr=PCA(n_components=2)
X = ptr.fit_transform(fingerprint)
utr=UMAP()
Y = utr.fit_transform(fingerprint)
plt.figure()
sns.scatterplot(X[:,0],X[:,1], hue=clusters['ip']['<lambda>'], palette='Spectral', size=clusters['ip']['count'])
plt.figure()
sns.scatterplot(Y[:,0],Y[:,1], hue=clusters['ip']['<lambda>'], palette='Spectral', size=clusters['ip']['count'])

# #just for vizualization
# embedding = umap.UMAP(
#     n_neighbors=20,
#     min_dist=0.0,
#     n_components=2,
#     random_state=42,
#     metric='jaccard',
# ).fit_transform(vect.loc[subset.index])
#
# sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1],
#                 size=matches/(matches.max()*10), hue=labels, edgecolors='w', palette='Spectral')

# ordering = \
#     pd.DataFrame(
#     data=np.stack(
#     fingerprint.apply(inter_arrival, args=[0.2], axis=1)),
#     #lambda x: np.correlate(x, x, mode='full')[len(x):len(x)+16],
#     index=fingerprint.index)
#
# #ordering = ordering.apply(lambda x: x/max(x,axis=1))
# ordering.sort_values(by=list(ordering.columns), ascending=False, inplace=True)
#%%
ordering = pd.DataFrame(data=np\
        .stack(fingerprint\
        .apply(inter_arrival, args=[0.0], axis=1)),
    #lambda x: np.correlate(x, x, mode='full')[len(x):len(x)+16],
    index=fingerprint.index)

#ordering = ordering.apply(lambda x: x/max(x,axis=1))
ordering.sort_values(by=list(ordering.columns), ascending=False, inplace=True)

#%%
# %matplotlib qt
#
# plt.figure()
# sns.heatmap(data=fingerprint.iloc[ordering.index, :])