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


        coded = ''
        for x in ips:
            coded += x + ','

        response = requests.post(self.base+'/ip/bulk/', data=coded[:-1], headers=self.headers)

        rep = [float(x) for x in list(response.text.split())]

        return pd.Series(data=rep, index=ips)

    #TODO fixme - what if there are no responses ?
    def ip_req(self, ips):

        ips = [x.strip(' ') for x in ips]

        ip_info = {}
        for ip in ips:
            resp = requests.get(self.base+'/ip/'+ip, headers=self.headers)
            if resp.status_code == 200:
                ip_info[ip] = json.loads(resp.text)
            else:
                ip_info[ip] = np.nan


        df = pd.DataFrame(columns=('geo', 'reputation', 'ipblock'))

        try:
            df = pd.DataFrame(data=ip_info).T.dropna(how='all')

            if len(df) > 0:
            #try:
                df['geo'] = df['geo'].apply(lambda x: x.get('ctry', None))
                #except KeyError:
                ##    print(f'{df.geo}\nIs missing "ctry" key', file=sys.stderr)

                df['reputation'] = df.fmp.apply(lambda x: x.get('general', None))

            #df.sort_values(inplace=True, by='reputation', ascending=False)
        except ValueError:
            pass

        return df


# might as well be considered as test
if __name__ == '__main__':
    ips=['192.168.0.1', '211.205.95.2']
    acc=NerdC()
    res=acc.ip_req(ips)



