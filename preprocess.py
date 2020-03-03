import json
import os
import datetime
import pandas as pd
import numpy as np

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
    time_marked = x.get('EventTime', '')
    if time_marked is '':
        time_marked = x.get('DetectTime', '')
    if time_marked is '':
        raise ValueError('Failed to extract EventTime or DetectTime timestamp')

    fmt = '%Y-%m-%d'

    if 'T' in time_marked:
        fmt = fmt+'T'
    else:
        fmt = fmt+' '

    fmt = fmt+'%H:%M:%S'
    if '.' in time_marked:
        fmt = fmt+'.%f'
    fmt = fmt+'%z'

    if time_marked[-1]=='Z':
        time_marked = time_marked.rstrip('Z')+'+0000'

    try:
        timestamp = int(datetime.datetime.strptime(time_marked, fmt).timestamp())
    except ValueError:
        raise

    #if window['min'] < 0:
    #    window['min'] = timestamp
    #elif timestamp < window['min']:
    #    window['min'] = timestamp
    #if window['max'] < timestamp:
    #    window['max'] = timestamp

    return timestamp


def get_series(evtsAt, length):
    length = np.int(length)
    vector = np.zeros(length, dtype=np.double)
    np.add.at(vector, np.array(evtsAt).astype(np.int), 1)
    #return pd.SparseArray(s, fill_value=0)
    return vector


def get_bin_series(evtsAt, length):
    length = np.int(length)
    vector = np.zeros(length, dtype=np.double)
    np.add.at(vector, np.array(evtsAt).astype(np.int), 1)
    #return pd.SparseArray(s, fill_value=0)
    return vector > 0

#Could be done better
def count_blocks(lst):
    last_val = lst[0]
    sum_blocks = 0
    for x in lst:
        if x > 0 and x != last_val:
            sum_blocks += 1
        last_val = x
    return sum_blocks


#Preprocess to time series of events, no features
def preprocess(file_path, silent=True):
    csv_str = "ip,timestamp,origin,type,line\n" #line is a bit misleading

    evt_types = {}
    origins = {}

    origins_n = -1
    evt_types_n = -1

    proc = 0
    line_num = 0

    signs={}
    openf = open

    if file_path[-3:] == '.gz':
        import gzip
        openf = gzip.open

    with openf(file_path, 'r') as data:
        lst = 0
        curr = 0
        line = data.readline()
        while line:
            #To build line index for full event retrieval, read event with readline()
            curr = lst
            lst = data.tell()

            line_num = line_num + 1
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

                evt_type = evt_types.get(category, evt_types_n+1)
                if evt_type > evt_types_n:
                    evt_types[category] = evt_types_n+1
                    evt_types_n = evt_types_n+1

                ip = extract_ip(x)

                timestamp = extract_time(x)

                csv_str += f"{ip},{timestamp},{origin},{evt_type},{curr}\n"#.format(ip, timestamp, origin, evt_type, curr)

                proc += 1

            except ValueError as err:
                if not silent:
                    print(err, end=', ')
                    print('while processing line {}'.format(line_num))
                pass
            finally:
                #if linenu > 1000: break
                pass

            line = data.readline()

    # with open('./data/Nodes.txt', 'w') as f:
    #     for key, val in signs.items():
    #         print('{}:{}'.format(key,val), file=f)

    res = pd.read_csv(StringIO(csv_str)) # it is faster the appending to data frame :)

    dt_origin = pd.CategoricalDtype(list(origins.keys()), ordered=True)
    dt_type = pd.CategoricalDtype(list(evt_types.keys()), ordered=True)

    res['origin'] = pd.Series(pd.Categorical.from_codes(codes=res['origin'].values, dtype=dt_origin))
    res['type'] = pd.Series(pd.Categorical.from_codes(codes=res['type'].values, dtype=dt_type))

    #if not silent:
    print('Processed {} % of events'.format(100 * proc / line_num))

    return res

def run_prep(file_path, prep_storage_dir):
    # file_path = './data/yyyy-mm-dd.idea'
    # where to store

    df = preprocess(file_path)

    path_str_list = os.path.split(file_path)
    file_name, suffix = path_str_list[1].split('.')

    file_path = prep_storage_dir + '/' + file_name
    df.to_pickle(file_path + '.pcl')

    return df


if __name__ == '__main__':

    src = sys.argv[1]
    dst = sys.argv[2]
    if dst == '':
        dst = './data'

    working_dir = os.fsencode(src)

    for file in os.listdir(working_dir):
        file_name = os.fsdecode(file)

        if file_name.endswith(".gz") or file_name.endswith(".idea"):
            file_path = os.path.join(os.fsdecode(working_dir), file_name)
            res = run_prep(file_path, dst)
        else:
            continue
