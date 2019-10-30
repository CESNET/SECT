import json
import os
import datetime
import pandas as pd
import numpy as np

#import sklearn
#from sklearn.model_selection import train_test_split
#from fastdtw import fastdtwhrome
#from dtaidistance import dtw
#from dtaidistance import dtw_visualisation as dtwvis

import gzip
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO



# Helper functions
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
def preprocess(datafile):
    csv_str = "ip,timestamp,origin,type,line\n" #line is a bit misleading

    etypes={}
    origins={}

    origins_n = -1
    etypes_n = -1

    proc = 0
    linenu = 0

    signs={}
    openf = open

    if datafile[-3:] == '.gz':
        import gzip
        openf = gzip.open

    with openf(datafile, 'r') as data:
        lst = 0
        curr = 0
        line = data.readline()
        while line:
            #To build line index for full event retrieval, read event with readline()
            curr = lst
            lst = data.tell()

            linenu = linenu + 1
            x = json.loads(line)

            try:
                #name = str(x['Node'][0]['Name'])
                name = str(x.get('Node', ['None']))
                category = str(x.get('Category', ['None']))

                #val = signs.get(str(x['Node']), 0)
                #signs[str(x['Node'])] = val+1

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

            line = data.readline()

    # with open('./data/Nodes.txt', 'w') as f:
    #     for key, val in signs.items():
    #         print('{}:{}'.format(key,val), file=f)

    res = pd.read_csv(StringIO(csv_str))

    dtOrigin = pd.CategoricalDtype(list(origins.keys()), ordered=True)
    dtType = pd.CategoricalDtype(list(etypes.keys()), ordered=True)

    res['origin'] = pd.Series(pd.Categorical.from_codes(codes=res['origin'].values, dtype=dtOrigin))
    res['type'] = pd.Series(pd.Categorical.from_codes(codes=res['type'].values, dtype=dtType))


    print('Processed {} % of events'.format(100 * proc / linenu))
    return res


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


def run_prep(filename, dst):
    # filename = './data/yyyy-mm-dd.idea'

    res = preprocess(filename)

    dirfile = os.path.split(filename)
    name, suffix = dirfile[1].split('.')

    filename = dst + '/' + name
    res.to_pickle(filename + '.pcl')

    return res


if __name__ == '__main__':

    src = sys.argv[1]
    dst = sys.argv[2]
    if dst == '':
        dst = './data'

    directory = os.fsencode(src)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        if filename.endswith(".gz") or filename.endswith(".idea"):
            fullpath = os.path.join(os.fsdecode(directory), filename)
            res = run_prep(fullpath, dst)
        else:
            continue
