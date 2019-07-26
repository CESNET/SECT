import json
import datetime
import pandas as pd

#%%
def dataset_write(ip_vect,org,fname):

    for k in ip_vect.keys():
        ip_vect[k] = sorted(set(ip_vect[k]))

    inv_org = {v: k for k, v in org.items()}
    fmt = '%Y-%m-%d %H:%M:%S%z'

    with open('./data/'+fname, 'w') as f:
        for key, val in ip_vect.items():
            print('{}\'{}\': ['.format('{',key),file=f,end='')
            for pair in val:
                print('(\'{}\',\'{}\'),'.format(
                    inv_org[pair[1]],
                    datetime.datetime.fromtimestamp(pair[0]).strftime(fmt)),file=f,end='')
            print(']}',file=f)

#    with open('./data/'+fname, 'w') as f:
#        for key in ip_vect:
#            json.dump({key:str(ip_vect[key])}, f)
#            print('',file=f)
#%%
def extract_set(f):

    origins = set()
    for line in f:
        X = json.loads(line)
        origins.add(X['Node'][0]['Name'])
    return origins
#%%
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
#%%

def preprocess(datafile):

    with open('./data/week03_sorted.ip','r') as ips:
        ip_vect={ip.strip(): [] for ip in ips}

    with open('./data/origins.txt','r') as org:
        lst=json.load(org)

    org={name: val for (name, val) in zip(lst, range(0,len(lst)))}

    feat=pd.DataFrame(columns=['ip','t_fisrt']+lst)
    feat['ip']=pd.Series(ip_vect.keys())

    with open('./data/'+datafile) as data:
        cnt = 0
        proc = 0
        for line in data:
            X = json.loads(line)
            if 'Source' in X.keys():
                ip = X['Source'][0].get('IP4', [''])[0]
                if ip is '':
                    ip = X['Source'][0].get('IP6', [''])[0]
                if ip is '':
                    cnt = cnt + 1
                    continue

                occ_time = X.get('EventTime', '')
                if occ_time is '':
                    occ_time = X.get('DetectTime', '')
                if occ_time is '':
                    cnt = cnt + 1
                    continue

                # occ_time2=occ_time[0:19]

                fmt = '%Y-%m-%d %H:%M:%S%z'

                if 'T' in occ_time:
                    if '.' in occ_time:
                        fmt = '%Y-%m-%dT%H:%M:%S.%f%z'
                    else:
                        fmt = '%Y-%m-%dT%H:%M:%S%z'
                elif '.' in occ_time:
                    fmt = '%Y-%m-%d %H:%M:%S.%f%z'

                try:
                    occ_time2 = datetime.datetime.strptime(occ_time, fmt)
                except ValueError:
                    cnt = cnt + 1
                    continue

                event=X['Node'][0]['Name']
                mark=org.get(event,'')
                ip_vect[ip].append(tuple([mark, int(occ_time2.timestamp())]))

                feat.loc[feat['ip'] == ip, event] = feat.loc[feat['ip'] == ip, event]+1

                proc = proc + 1

            if proc > 10000:
                break

    print('Processed {} % of events'.format(100*proc/cnt))
    return [ip_vect, feat]
#%%
def preprocess2(datafile):

#Todo extract if not available
    with open('./data/week03_sorted.ip','r') as ips:
        ip_vect={ip.strip(): [] for ip in ips}

    with open('./data/origins.txt','r') as org:
        lst=json.load(org)

    org={name: val for (name, val) in zip(lst, range(0,len(lst)))}

    # feat=pd.DataFrame(columns=['ip','t_fisrt']+lst)
    # feat['ip']=pd.Series(ip_vect.keys())

    with open('./data/'+datafile) as data:
        cnt = 0
        proc = 0
        for line in data:
            cnt=cnt+1
            X = json.loads(line)
            if 'Source' in X.keys():
                ip = X['Source'][0].get('IP4', [''])[0]
                if ip is '':
                    ip = X['Source'][0].get('IP6', [''])[0]
                if ip is '':
                    continue

                occ_time = X.get('EventTime', '')
                if occ_time is '':
                    occ_time = X.get('DetectTime', '')
                if occ_time is '':
                    continue

                # occ_time2=occ_time[0:19]

                fmt = '%Y-%m-%d %H:%M:%S%z'

                if 'T' in occ_time:
                    if '.' in occ_time:
                        fmt = '%Y-%m-%dT%H:%M:%S.%f%z'
                    else:
                        fmt = '%Y-%m-%dT%H:%M:%S%z'
                elif '.' in occ_time:
                    fmt = '%Y-%m-%d %H:%M:%S.%f%z'

                try:
                    occ_time2 = datetime.datetime.strptime(occ_time, fmt)
                except ValueError:
                    continue

                event=X['Node'][0]['Name']
                mark=org.get(event,'')
                ip_vect[ip].append(tuple([int(occ_time2.timestamp()),mark]))

                #feat.loc[feat['ip'] == ip, event] = feat.loc[feat['ip'] == ip, event]+1

                proc = proc + 1

    print('Processed {} % of events'.format(100*proc/cnt))
    return [ip_vect, org]

#%%
if __name__=='__main__':

    [ip_vect,org] = preprocess2('week03.idea')
    dataset_write(ip_vect,org,'week03_sources.txt')

    #feat.head()