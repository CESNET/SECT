import json
import os.path
import datetime
import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
#from fastdtw import fastdtw
from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis

# Dimension reduction and clustering libraries


import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

def _dataset_write_old(fname, data):

    ip_vect = data['ips']

    for k in ip_vect.keys():
        ip_vect[k] = sorted(set(ip_vect[k]))

    origins = data['origins']
    fmt = '%Y-%m-%d %H:%M:%S%z'

    f = open(fname + 'sources.txt', 'w')
    g = open(fname + 'detected_ips.txt', 'w')
    for key, val in ip_vect.items():
        print('{}\'{}\': ['.format('{', key), file=f, end='')
        print('{}\'{}\': ['.format('{', key), file=g, end='')

        for pair in val:
            timestr = datetime.datetime.fromtimestamp(pair[0]).strftime(fmt)
            print('(\'{}\',\'{}\')'.format(origins[pair[1]], timestr),
                  file=f, end=',')
            print('\'{}\''.format(timestr), file=g, end=',')

        print(']}', file=f)
        print(']}', file=g)
    f.close()
    g.close()

def _extract_features(ip_vect, org, type):

    feat={}
    for key, val in ip_vect.items():
        if len(val)>0:
            tmp=dict(t_start=val[0][0],duration=val[-1][0]-val[0][0]+1,count=len(val))
            org_counts = [0 for x in range(0,len(org))]
            type_counts = [0 for x in range(0,len(type))]

            for vect in val:
                org_counts[vect[1]]=org_counts[vect[1]]+1
                type_counts[vect[2]]=type_counts[vect[2]]+1

            org_feat = {org[v]: org_counts[v] for v in range(0,len(org))}
            type_feat = {type[v]: type_counts[v] for v in range(0,len(type))}
            tmp.update(org_feat)
            tmp.update(type_feat)
            feat[key]=tmp
    feat=pd.DataFrame(feat).transpose()
    return feat

#%% Helper functions

def extract_ip(x):
    if 'Source' in x.keys():
        ip = x['Source'][0].get('IP4', [''])[0]
        if ip is '':
            ip = x['Source'][0].get('IP6', [''])[0]
        if ip is '':
            raise ValueError('Failed to extract IP address')
    else:
        raise ValueError('Failed to access Source record while extracting IP address')
    return ip


def extract_time(x): #window):
    occ_time = x.get('EventTime', '')
    if occ_time is '':
        occ_time = x.get('DetectTime', '')
    if occ_time is '':
        raise ValueError('Failed to extract EventTime or DetectTime timestamp')

    fmt = '%Y-%m-%d'

    if 'T' in occ_time:
        fmt=fmt+'T'
    else:
        fmt=fmt+' '

    fmt=fmt+'%H:%M:%S'
    if '.' in occ_time:
        fmt=fmt+'.%f'
    fmt=fmt+'%z'

    try:
        timestamp = int(datetime.datetime.strptime(occ_time, fmt).timestamp())
    except ValueError:
        raise

    #if window['min'] < 0:
    #    window['min'] = timestamp
    #elif timestamp < window['min']:
    #    window['min'] = timestamp
    #if window['max'] < timestamp:
    #    window['max'] = timestamp

    return timestamp

# Preprocess to time series of events, no features
def preprocess_lite(datafile):
    csv_str = "ip,timestamp,origin,type,line\n" #line is a bit misleading

    etypes={}
    origins={}

    origins_n = -1
    etypes_n = -1

    proc = 0
    linenu = 0

    with open(datafile) as data:
        lst = 0
        curr = 0
        for line in data:

            #To build line index for full event retrieval, read event with readline()
            curr=lst
            lst=data.tell()

            linenu = linenu + 1
            x = json.loads(line)

            try:
                name = x['Node'][0]['Name']
                category = x.get('Category', ['None'])[0]

                origin = origins.get(name, origins_n+1)
                if origin > origins_n:
                    origins[name] = origins_n+1
                    origins_n = origins_n+1

                etype = etypes.get(category, etypes_n+1)
                if etype > etypes_n:
                    etypes[category] = etypes_n+1
                    etypes_n = etypes_n+1

                ip = extract_ip(x)

                timestamp = extract_time(x)

                csv_str += ("{},{},{},{},{}\n".format(ip, timestamp, origin, etype, curr))

                proc += 1

            except ValueError as err:
                print(err, end=', ')
                print('while processing line {}'.format(linenu))
                pass
            finally:
                #if linenu > 1000: break
                pass

    res = pd.read_csv(StringIO(csv_str))

    dtOrigin = pd.CategoricalDtype(list(origins.keys()), ordered=True)
    dtType = pd.CategoricalDtype(list(etypes.keys()), ordered=True)

    res['origin'] = pd.Series(pd.Categorical.from_codes(codes=res['origin'].values, dtype=dtOrigin))
    res['type'] = pd.Series(pd.Categorical.from_codes(codes=res['type'].values, dtype=dtType))


    print('Processed {} % of events'.format(100 * proc / linenu))
    return res

# Preprocess to series of events with features marked
def preprocess(datafile):

    res = {}
    origins = {}
    types = {}
    ip_vect = {}
    window = dict(min=-1, max=0)

    origins_n = -1
    types_n = -1

    total = 0
    proc = 0

    origin_stats={}

    #df=pd.DataFrame(columns=['ip', 'origin', 'type', 'time'])

    #da=open(datafile+'_pd.csv', 'w')
    csv_str = ""
    #print("ip,origin,type,time",file=da)
    with open(datafile) as data:
        category = set()
        node = set()
        for line in data:
            x = json.loads(line)

            for cat in x.get('Category', []):
                category.add(cat)
            for n in x.get('Node', []):
                node.add(n['Name'])

            #total += 1
            #if total > 1000: break
        #index dictionaries
        toIdx = dict(zip(['ip', 'time', 'line']+sorted(category)+sorted(node), range(-3,len(category)+len(node)-3)))
        #nodes = dict(zip(sorted(node),range(2+len(category),2+len(category)+len(node))))
        for s in toIdx.keys():
            csv_str+=s+','
        csv_str+='\n'

        data.seek(0)
        for line in data:
            total = total+1
            x = json.loads(line)

            vect = np.zeros(len(toIdx)-3)
            try:
                name = x['Node'][0]['Name']
                vect[toIdx[name]] += 1
                category = x.get('Category', ['None'])
                for c in category:
                    vect[toIdx[c]] += 1

                ip = extract_ip(x)

                timestamp = extract_time(x, window)



                #index is now actualy line number in .idea file
                #df.at[total,:]={'ip': ip,'origin': mark,'type': type, 'time': timestamp}
                csv_str+=("{},{},{},{}\n".format(ip, timestamp, total, str(vect)[1:-1]))

                proc = proc+1

            except ValueError as err:
                print(err, end=', ')
                print('while processing line {}'.format(total))
                pass
            finally:
                #if total > 1000: break
                pass

    #res['timespan'] = window
    #res['origins'] = {v: k for k, v in origins.items()}
    #res['types'] = {v: k for k, v in types.items()}
    #res['origin_stats'] = origin_stats
    res = pd.read_csv(StringIO(csv_str))



    #feat = pd.DataFrame(columns=(tuple(['ip'] + list(dfile['types'].values()) + ['count'] + ['duration'])))

    #feat.ip = B.ip
    #feat.fillna(0, inplace=True)

    #for key, val in dfile['types'].items():
    #    feat[val] = B['type'] == int(key)
        # Bfeat = B.groupby('ip')

    # Todo do i really need this? This functionallity should be covered in pandas dataframe
    # for key, vect in res['ips'].items():
    #     tmp = sorted(vect)
    #     aggreg = []
    #     last = tmp[0]
    #     cnt = 1
    #     for val in next(tmp):
    #         if val == last:
    #             cnt = cnt+1
    #         else:
    #             aggreg.append(tuple(last, cnt))
    #             cnt = 1
    #             last = val

    print('Processed {} % of events'.format(100*proc/total))
    return res


def get_aggregated(df, tfrom=0, tto=0):
    #Todo make sparse array in dataframe
    if tfrom is not 0:
        tfrom = datetime.datetime.fromisoformat(tfrom).timestamp()
    if tto is not 0:
        tto = datetime.datetime.fromisoformat(tto).timestamp()

    A = df.copy()
    if tfrom > 0:
        A = A[df['timestamp'] >= tfrom]
    if tto > 0:
        A = A[A['timestamp'] < tto]

    #if agg_win > 1:
    #    A['timestamp'] = np.int32(A['timestamp']/agg_win)*agg_win

    A['timestamp'] = A['timestamp'] - tfrom
    series = {
        'timestamp': [('list', list),
                ('count', 'count'),
                 #('sparse', np.zeros(tto-tfrom))
                ]
    }

    sp_signal=A.groupby('ip').agg([series])

    return sp_signal


def get_series(evtsAt, length):
    length = np.int(length)
    s = np.zeros(length, dtype=np.double)
    np.add.at(s, np.array(evtsAt).astype(np.int), 1)
    #return pd.SparseArray(s, fill_value=0)
    return s


def get_bin_series(evtsAt, length):
    length = np.int(length)
    s = np.zeros(length, dtype=np.double)
    np.add.at(s, np.array(evtsAt).astype(np.int), 1)
    #return pd.SparseArray(s, fill_value=0)
    return s > 0


def count_blocks(lst):
    v = lst[0]
    res = 0
    for x in lst:
        if x > 0 and x != v:
            res += 1
        v=x
    return res


def run_prep(fname):
    filename = fname
    # filename = './data/yyyy-mm-dd.idea'

    res = preprocess_lite(filename)

    dirfile = os.path.split(filename)
    name, suffix = dirfile[1].split('.')

    filename = dirfile[0] + '/' + name

    # dataset_write_old(filename, res)
    # dump raw preprocessed data
    #with open(filename + '_.csv', 'w') as f:
    #    print(res.to_csv(), file=f)
    res.to_pickle(filename+'.pcl')
    return res

if __name__ == '__main__':

#   batch={}
#   for day in range(11,18):
#       filename = './data/2019-03-{}.idea'.format(day)
#       #filename = './data/week03.idea'

#       res = preprocess(filename)

#       dirfile = os.path.split(filename)
#       name, suffix = dirfile[1].split('.')

#       filename = dirfile[0]+'/'+name

#       #dataset_write_old(filename, res)
#       #dump raw preprocessed data
#       with open(filename + '_prep.json', 'w') as f:
#           json.dump(res, f)

#       batch[os.path.split(filename)[1]] = res

    res=run_prep('./data/2019-08-01.idea')



