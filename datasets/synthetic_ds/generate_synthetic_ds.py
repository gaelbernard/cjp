import pandas as pd
import numpy as np
from random import expovariate


def simulatePoissonProcess(size, rate):
    return [expovariate(1/rate) for _ in range(size)]


'''
Parse synthetic dataset produced by PROM (csv format)
and generate "journey_id" which consist of randomly concatenated traces.

The ultimate goal is to build an algorithm that "deconcatenate" the big traces
== Finding back the initial trace by using only the columns "timestamp" and "event".

We then use a Poisson process to simulate the time between events.
When we jump from one process to another, we add a delay.
The bigger the delay, the easier it is to correctly "deconcatenate"
'''
df = pd.read_csv('/Users/gbernar1/Dropbox/PhD_cloud/01-Research/02-present/Customer_Journey_Mapping/28-CJ_with_PM/synthetic_dataset/pm.csv')\
    .drop(['startTime', 'completeTime', 'simulated:logProbability', 'concept:simulated'], axis=1)\
    .reset_index()
n_final_journeys = 10
rate = 1
df['journey_id'] = df['case'].map(
    pd.Series(np.random.randint(0, n_final_journeys, size=df['case'].nunique()), index=df['case'].unique()).to_dict()
)
df.sort_values(by=['journey_id', 'index'], inplace=True)
df['random_distributed_time'] = simulatePoissonProcess(df.shape[0], rate)
df['new_trace'] = df['case']!=df['case'].shift(1)
df['last_trace'] = df['case']!=df['case'].shift(-1)


for delay in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 1.0, 2.0]:
    c = df.copy()
    c['delay'] = 0

    c.loc[c['new_trace']==True, 'delay'] = delay
    c['random_distributed_time_delay'] = c['random_distributed_time']+c['delay']
    c['timestamp'] = c['random_distributed_time_delay'].cumsum()

    c['time_diff'] = c.groupby('journey_id')['timestamp'].shift(-1)-c['timestamp']
    c.drop(['index', 'new_trace', 'random_distributed_time_delay'], axis=1)
    print (c['case'].nunique())
    c.to_csv('synthetic_1_delay{}.csv'.format(round(delay,2)))
