import sys
import requests
import json
import pandas as pd
import numpy as np


class NerdC:
    def __init__(self):

        self.base = 'https://nerd.cesnet.cz/nerd/api/v1'
        self.token = 'RsUF9Xpdmg'


        self.headers = {
            'Authorization': '{}'.format(self.token),
            "Content-Type": "text/plain"
        }

    def bulk_req(self, ips):
        ips = [x.strip(' ') for x in ips]


        coded=''
        for x in ips:
            coded+=x+','

        response = requests.post(self.base+'/ip/bulk/', data=coded[:-1], headers=self.headers)

        rep = [float(x) for x in list(response.text.split())]

        return pd.Series(data=rep, index=ips)

    def ip_req(self, ips):

        ips = [x.strip(' ') for x in ips]
        ip_info = {}
        for ip in ips:
            resp = requests.get(self.base+'/ip/'+ip, headers=self.headers)
            if resp.status_code == 200:
                ip_info[ip] = json.loads(resp.text)
            else:
                ip_info[ip] = np.nan

        df = pd.DataFrame(data=ip_info).T.dropna(how='all')

        try:
            df['geo'] = df['geo'].apply(lambda x: x['ctry'])
        except KeyError:
            print(f'{df.geo}\nIs missing "ctry" key', file=sys.stderr)

        try:
            df['reputation'] = df.fmp.apply(lambda x: x['general'])
        except KeyError:
            print(f'{df.reputation}\nIs missing "reputation" key', file=sys.stderr)
        #df.sort_values(inplace=True, by='reputation', ascending=False)

        return df


# might as well be considered as test
if __name__ == '__main__':
    ips=['192.168.0.1', '211.205.95.2']
    acc=NerdC()
    res=acc.ip_req(ips)



