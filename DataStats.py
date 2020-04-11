import numpy as np
import pandas as pd
import datetime
from utils import *

from preprocess import *


import TemporalClusterer


from IPython import get_ipython


if __name__ == '__main__':

    ipy = get_ipython()
    if ipy is not None:
        ipy.run_line_magic('matplotlib', 'qt')

    # Load preprocessed files
    days = int(sys.argv[4])
    intervals = list(pd.date_range(sys.argv[2], sys.argv[3]))
    result = pd.DataFrame()

    tmp=pd.DataFrame()
    for idx in range(0, len(intervals), days):
        #print(f"Processing files for {intervals[idx].date().isoformat()} - {intervals[idx+days-1].date().isoformat()}")
        (df, file_list) = load_files(sys.argv[1], intervals[idx].date().isoformat(), intervals[idx+days-1].date().isoformat())

        #print("Clustering")
        tc = TemporalClusterer.TemporalClusterer(min_events=0, min_activity=0, max_activity=1, aggregation=sys.argv[4])
        dft = tc.transform(df, [])
        dft['index'] = idx
        result = pd.concat([result, dft])
        tmp=result.groupby(result.index)['count'].agg('sum')
        print([sum(tmp),len(tmp)])

    act_cnts = pd.DataFrame(data=result.groupby('index')['activity'].agg(list).apply(lambda x: pd.Series(x).value_counts()))
    evt_cnts = pd.DataFrame(data=result.groupby('index')['count'].agg(list).apply(lambda x: pd.Series(x).value_counts()))

    series = pd.DataFrame(data=np.stack(result.series), index=result.index)
    series['ip'] = series.index
    series['activity'] = result.activity
    groups = series.loc[result['index'] == 0, :].groupby(list(range(0, 96))).agg(['count', 'min']).values
    groups = pd.DataFrame(data=groups, columns=('count', 'ip', 'count2', 'activity'))



approx=pd.DataFrame([val.replace(',','.').split('\t') for val in """2	0,058767512	0,01326454	2,90964E-05	8,65661E-08	3,76475E-10
3	0,109298033	0,00078717	7,42658E-08	1,20083E-11	3,44333E-15
4	0,152451457	3,50339E-05	1,42162E-10	1,24927E-15	2,36192E-20
5	0,170107603	1,24733E-06	2,17696E-13	1,0397E-19	1,29606E-25
6	0,158167609	3,70064E-08	2,77792E-16	7,21037E-24	5,92633E-31
7	0,126051352	9,41041E-10	3,03826E-19	4,28593E-28	2,32264E-36
8	0,087895865	2,09378E-11	2,90751E-22	2,22907E-32	7,96473E-42
9	0,054477827	4,14079E-13	2,47314E-25	1,03046E-36	2,42767E-47
10	0,03038761	7,36989E-15	1,89322E-28	4,28712E-41	6,65937E-53
11	0,015408613	1,19242E-16	1,31748E-31	1,6214E-45	1,66061E-58
12	0,007161845	1,76845E-18	8,40387E-35	5,62094E-50	3,79574E-64
13	0,003072608	2,42089E-20	4,94808E-38	1,79866E-54	8,0084E-70
14	0,001224017	3,07721E-22	2,70516E-41	5,34425E-59	1,56889E-75
15	0,00045508	3,65055E-24	1,38028E-44	1,48199E-63	2,86854E-81
16	0,000158614	4,05988E-26	6,60234E-48	3,85262E-68	4,9168E-87
17	5,20296E-05	4,24935E-28	2,97222E-51	9,42587E-73	7,93154E-93
18	1,61182E-05	4,20041E-30	1,26364E-54	2,17794E-77	1,20835E-98
19	4,73028E-06	3,93335E-32	5,08943E-58	4,76731E-82	1,7439E-104
20	1,31875E-06	3,49896E-34	1,94724E-61	9,91303E-87	2,391E-110""".split('\n')],dtype=np.float)
approx.index=approx[0]
approx.columns+=1
ax = approx.loc[:,2:].plot(xticks=approx.index, title='Probability estimate for n-fold random matches in data')
ax.set_xlabel('n_tuple')
ax.set_ylabel("probability")
print(approx.loc[:,2:].sum().to_latex())

sumdf=pd.DataFrame()
for val in range(1,97):
    tmp=pd.DataFrame(np.histogram(groups.loc[groups[('activity','min')] == val, ('activity','count')],
                                 bins=list(range(1, 22, 1))+[10000]))
    sumdf=pd.concat([sumdf,tmp])
sumdf=sumdf.loc[0,:]
sumdf.index=list(range(1,97))
sumdf.columns=list(range(1, 22, 1))+[10000]
sumdf=sumdf.iloc[:,:-1]

act_ips = result.groupby('activity')['count'].agg('sum')
observed_pst=sumdf.T/(sumdf.T.sum())
ax=observed_pst.loc[2:,2:7].plot(xticks=observed_pst.index)
ax.set_title('Observed probability density for n-fold matches in data')
ax.set_xlabel('n_tuple')
ax.set_ylabel("probability")
print(observed_pst.loc[2:,2:7].sum().to_latex())