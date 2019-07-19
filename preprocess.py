import json
import datetime
import pandas as pd
import numpy as np

#%%
def ip_dump(ip_vect, fname, feat):
    for k in ip_vect.keys():
        ip_vect[k] = sorted(ip_vect[k])

    with open('data/week03_detected_ips.txt', 'w') as f:
        for key in ip_vect:
            json.dump({key: ip_vect[key]}, f)
            print('',file=f)

#%%
if __name__=='__main__':

    ip_vect = {}
    ip_feat = {}
    ip_src = {}
    with open('data/week03.idea') as data:
        cnt = 0
        proc = 0
        for line in data:
            X=json.loads(line)
            if 'Source' in X.keys() and ('IP4' in X['Source'][0].keys() or 'IP6' in X['Source'][0].keys()):
                ip = X['Source'][0].get('IP4', [''])[0]
                if ip is '':
                    ip = X['Source'][0]['IP6'][0]

                occ_time = X.get('EventTime', '')
                if occ_time is '':
                    occ_time = X['DetectTime']

                #occ_time2=occ_time[0:19]
                if 'T' in occ_time:
                    occ_time2=datetime.datetime.strptime(occ_time, '%Y-%m-%dT%H:%M:%S%z')
                else:
                    occ_time2=datetime.datetime.strptime(occ_time, '%Y-%m-%d %H:%M:%S%z')

                if ip not in ip_vect.keys():
                    ip_vect[ip] = set()
                    ip_feat[ip] = 0
                ip_vect[ip].add(int(occ_time2.timestamp()))
                ip_feat[ip] = ip_feat[ip] + 1

                proc=proc + 1
            cnt = cnt + 1

#%%
#ip_dump(ip_dict)

d = ip_vect.copy()
for key in d:

    vec = d[key]
    dur = vec[-1]-vec[0]+1
    cnt = len(vec)
    start_t = vec[0]
    evt_cnt = 0

    d[key] = { }

with open('data/week03.idea','r') as data:
    cnt = 0
    proc = 0
    cats= set()
    for line in data:
        X=json.loads(line)
        cats.add(X['Category'][0])

with open('data/week03.idea','r') as data:
    cnt = 0
    proc = 0
    origins= set()
    for line in data:
        X=json.loads(line)
        origins.add(X['Node'][0]['Name'])