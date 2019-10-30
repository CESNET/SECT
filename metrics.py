import pandas as pd
import numpy as np

#tmp = np.sum(np.bitwise_and(data.iloc[x, 2], data.iloc[y, 2])) / max(np.sum(data.iloc[x, 2]), np.sum(data.iloc[y, 2]))
#%% md
Small performance survey
#%%
%timeit tmp = np.sum(np.bitwise_and(data.iloc[x,2], data.iloc[y,2]))/max(np.sum(data.iloc[x,2]),np.sum(data.iloc[y,2]))
#%%
%timeit tmp = np.sum(np.bitwise_and(data.iloc[x,2], data.iloc[y,2]))/max(data.iloc[x,2],data.iloc[y,2])
#%%
%timeit tmp = np.sum(np.multiply(data.iloc[x,2], data.iloc[y,2]))/max(np.sum(data.iloc[x,2]),np.sum(data.iloc[y,2]))
#%%
%timeit tmp = np.sum(np.bitwise_and(vect.iloc[x,:], vect.iloc[y,:]))/max(np.sum(vect.iloc[x,:]),np.sum(vect.iloc[y,:]))

