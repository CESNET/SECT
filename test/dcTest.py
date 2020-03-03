import dCollector
import pandas as pd

def test_filterFlows():

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

    tFrom = 1581727800
    tTo = 1581729000

    dc = dCollector.dc()
    df = dc.filterFlows(filter, tFrom, tTo)

    print(df.head())
    return df

if __name__=='__main__':
    a = test_filterFlows()

