import json
import os.path
import datetime
import pandas as pd


def dataset_write_old(fname, data):
    ip_vect=data['ips']

    for k in ip_vect.keys():
        ip_vect[k] = sorted(set(ip_vect[k]))

    origins=data['origins']
    fmt = '%Y-%m-%d %H:%M:%S%z'

    f = open(fname+'sources.txt', 'w')
    g = open(fname+'detected_ips.txt', 'w')
    for key, val in ip_vect.items():
        print('{}\'{}\': ['.format('{', key), file=f, end='')
        print('{}\'{}\': ['.format('{', key), file=g, end='')

        for pair in val:
            timestr=datetime.datetime.fromtimestamp(pair[0]).strftime(fmt)
            print('(\'{}\',\'{}\')'.format(origins[pair[1]], timestr),
                  file=f, end=',')
            print('\'{}\''.format(timestr), file=g, end=',')

        print(']}', file=f)
        print(']}', file=g)
    f.close()
    g.close()

def extract_features(ip_vect, org):

    inv_org = {v: k for k, v in org.items()}
    feat={}
    for key, val in ip_vect.items():
        if len(val)>0:
            tmp=dict(t_start=val[0][0],duration=val[-1][0]-val[0][0]+1,count=len(val))
            org_counts = [0 for x in range(0,len(org))]
            for pair in val:
                org_counts[pair[1]]=org_counts[pair[1]]+1
            org_feat = {inv_org[v]: org_counts[v] for v in range(0,len(org))}
            tmp.update(org_feat)
            feat[key]=tmp
#Todo repair so it return ips as indexes
    feat=pd.DataFrame(feat)
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


def extract_time(x, window):
    occ_time = x.get('EventTime', '')
    if occ_time is '':
        occ_time = x.get('DetectTime', '')
    if occ_time is '':
        raise ValueError('Failed to extract EventTime or DetecTime timestamp')

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

    if window['min'] < 0:
        window['min'] = timestamp
    elif timestamp < window['min']:
        window['min'] = timestamp
    if window['max'] < timestamp:
        window['max'] = timestamp

    return timestamp


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

    with open(datafile) as data:
        for line in data:
            total = total+1
            x = json.loads(line)
            try:
                name = x['Node'][0]['Name']
                category = x.get('Category', ['None'])[0]

                # Gather per origin statistics
                n = origin_stats.get(name, {})
                val = n.get(category,0)
                n.update({category: val+1})
                origin_stats[name] = n

                ip = extract_ip(x)

                timestamp = extract_time(x, window)

                mark = origins.get(name, origins_n+1)
                if mark > origins_n:
                    origins[name] = origins_n+1
                    origins_n = origins_n+1

                type = types.get(category, types_n + 1)
                if type > types_n:
                    types[category] = types_n + 1
                    types_n = types_n + 1
                    
                # Encode reduced data
                v = ip_vect.get(ip, [])
                v.append(tuple([timestamp, mark, type]))
                ip_vect[ip] = v

                proc = proc + 1
            except ValueError as err:
                print(err, end=', ')
                print('while processing line {}'.format(total))
                continue

    res['timespan'] = window
    res['origins'] = {v: k for k, v in origins.items()}
    res['types'] = {v: k for k, v in types.items()}
    res['origin_stats'] = origin_stats
    res['ips'] = ip_vect


    print('Processed {} % of events'.format(100*proc/total))
    return res


if __name__ == '__main__':

    batch={}
    for day in range(11,18):
        filename = './data/2019-03-{}.idea'.format(day)
        #filename = './data/week03.idea'

        res = preprocess(filename)

        dirfile = os.path.split(filename)
        name, suffix = dirfile[1].split('.')

        filename = dirfile[0]+'/'+name

        #dataset_write_old(filename, res)

        f=open(filename+'_prep.json', 'w')
        json.dump(res, f)

        batch[os.path.split(filename)[1]]=res
#%%
