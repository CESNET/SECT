from __future__ import with_statement #was supposed to be used to hide output stream while running

import pandas as pd
from fabric import Connection

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

import datetime

def dc_liberouter_filter_string(val):
    return 'ip in {}'.format(list(val)).replace('\'', '')

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
                    fields=('first', 'packets', 'bytes', 'srcip', 'dstip', 'srcport', 'dstport', 'flags',
                            'proto', 'duration'),
                    addr_conv='str', tcpflags_conv='none', proto_conv='none', duration_conv='none'):

        channels_s = ''
        for x in channels:
            channels_s = channels_s + dc.channels[x] + ' '

        fields_s = ''
        for x in fields:
            if fields=='':
                fields_s += x
            else:
                fields_s +=','+x

        command = f"fdistdump-ha {channels_s} " \
                  "/usr/lib64/mpich/bin/mpiexec  /usr/lib64/mpich/bin/fdistdump_mpich --num-threads 8 "\
                  " --output-rich-header " \
                  f"-f \"{filter}\" " \
                  f"-T {tFrom}\#{tTo} " \
                  f"-l {limit} " \
                  f"--output-format=csv --output-tcpflags-conv={tcpflags_conv} --output-addr-conv={addr_conv} "\
                  f"--output-proto-conv={proto_conv} " \
                  f"--output-duration-conv={duration_conv} --output-volume-conv=none " \
                  "--output-items=r --time-zone=\"Europe/Bratislava\" " \
                  f"--output-fields={fields_s}"



        #result = c.run('mpiexec -host dc,dc1,dc2,dc3 fdistdump_mpich -f any -l 10 ' \
        #               '/media/flow/original/safeguard/live/channels/nix_zikova/2019/10/12/lnf.20191012122000')
        #with self.c.hide('output'):
        print(command)
        result = self.c.run(f"export PATH={dc.path} && {command}", hide='stdout')
        return pd.read_csv(StringIO(result.stdout), sep=',')


