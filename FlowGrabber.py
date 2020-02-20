import pandas as pd
from fabric import Connection

import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO

filter = """ip in ['107.155.36.2',
 '119.28.164.101',
 '128.1.102.27',
 '128.1.122.18',
 '150.109.88.30',
 '150.109.90.105',
 '170.106.32.101',
 '175.97.130.20',
 '203.205.224.43',
 '211.152.144.95',
 '211.152.147.21',
 '23.248.179.21',
 '23.248.189.24',
 '47.91.154.29',
 '54.233.192.109']"""

tfrom=1581727800
tto=1581729000

channels = "live/channels/nix_zikova live/channels/nix_dctower live/channels/tis live/channels/geant "\
          "live/channels/amsix live/channels/sanet live/channels/aconet live/channels/pioneer "\
          "live/channels/other"


path="/usr/lib64/mpich/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/home/current/$USER/.local/bin:/home\
/current/$USER/bin"

command = f"fdistdump-ha {channels} " \
        "/usr/lib64/mpich/bin/mpiexec  /usr/lib64/mpich/bin/fdistdump_mpich --num-threads 8 --output-rich-header " \
        f"-f \"{filter}\" " \
        f"-T {tfrom}\#{tto} " \
        "-l 1000  --output-format=csv --output-tcpflags-conv=str --output-addr-conv=str --output-proto-conv=str " \
        "--output-duration-conv=str --output-volume-conv=none " \
        "--output-items=r --time-zone=\"Europe/Bratislava\"" \
        "--output-fields=first,packets,bytes,srcip,dstip,srcport,dstport,flags,proto,duration"


c = Connection(host='dc.liberouter.org', user='xstoff02', connect_kwargs={
                "key_filename": "C:\\Users\\Imrich\\.ssh\\dc"}
               )
#result = c.run('mpiexec -host dc,dc1,dc2,dc3 fdistdump_mpich -f any -l 10 ' \
#               '/media/flow/original/safeguard/live/channels/nix_zikova/2019/10/12/lnf.20191012122000')

result = c.run(f"export PATH={path} && {command}")

print(result)

df = pd.read_csv(StringIO(result.stdout),
                 sep='\s,')

#%%
import fabric

@fabric.tasks.task
def staging(ctx):
    ctx.name = 'test'
    ctx.user = 'xstoff02'
    ctx.host = 'benefizio.liberouter.org'
    ctx.connect_kwargs.key_filename = 'C:\\Users\\Imrich\\.ssh\\benefizio'
    return ctx

@fabric.tasks.task
def do_something_remote(ctx):
    with fabric.connection.Connection(ctx.host, ctx.user, connect_kwargs=ctx.connect_kwargs) as conn:
        conn.run('hostname')

#%%
from fabric import Connection
c = Connection(host='benefizio.liberouter.org',user='xstoff02',connect_kwargs={
        "key_filename": "C:\\Users\\Imrich\\.ssh\\dc",
    },)
result = c.run('ssh dc mpiexec -host dc,dc1,dc2,dc3 fdistdump_mpich -f any -l 10 '\
               '/media/flow/original/safeguard/live/channels/nix_zikova/2019/10/12/lnf.20191012122000')

result.stdout.strip() == 'Linux'
result.exited
result.ok
result.command
result.connection
result.connection.host
