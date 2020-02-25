import pandas as pd
from fabric import Connection

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import datetime

class dc:
    def __init__(self):
        self.c = Connection(host='dc', user='xstoff02')

    path = "/usr/lib64/mpich/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/current/$USER/.local/bin:/home"\
    "/current/$USER/bin"

    channels = dict(zip(['zikova', 'dctower', 'tis', 'geant', 'amsix', 'sanet', 'aconet', 'pioneer'],
                    "live/channels/nix_zikova live/channels/nix_dctower live/channels/tis live/channels/geant " \
                    "live/channels/amsix live/channels/sanet live/channels/aconet live/channels/pioneer " \
                    "live/channels/other".split(' ')))

    def filterFlows(self, filter, tFrom, tTo, agg=(),
                    channels=('zikova', 'dctower', 'tis', 'geant', 'amsix', 'sanet', 'aconet', 'pioneer'),
                    limit=10000,
                    fields=('first','packets','bytes','srcip','dstip','srcport','dstport','flags','proto','duration')):

        channels_s = ''
        for x in channels:
            channels_s = channels_s + dc.channels[x] + ' '

        fields_s = ''
        for x in fields:
            if fields=='':
                fields_s=x
            else:
                fields_s =','+x

        command = f"fdistdump-ha {channels_s} " \
                  "/usr/lib64/mpich/bin/mpiexec  /usr/lib64/mpich/bin/fdistdump_mpich --num-threads 8 --output-rich-header " \
                  f"-f \"{filter}\" " \
                  f"-T {tFrom}\#{tTo} " \
                  f"-l {limit} " \
                  "--output-format=csv --output-tcpflags-conv=str --output-addr-conv=str --output-proto-conv=str " \
                  "--output-duration-conv=str --output-volume-conv=none " \
                  "--output-items=r --time-zone=\"Europe/Bratislava\"" \
                  f"--output-fields={fields_s}"



        #result = c.run('mpiexec -host dc,dc1,dc2,dc3 fdistdump_mpich -f any -l 10 ' \
        #               '/media/flow/original/safeguard/live/channels/nix_zikova/2019/10/12/lnf.20191012122000')

        result = self.c.run(f"export PATH={dc.path} && {command}")
        return pd.read_csv(StringIO(result.stdout), sep=',')


