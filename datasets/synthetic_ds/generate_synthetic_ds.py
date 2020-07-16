import pandas as pd
import numpy as np
from random import expovariate

def simulatePoissonProcess(size, rate):
    return [expovariate(1/rate) for _ in range(size)]


'''
1) We first define a process model as shown in the Section "Evaluation using Synthetic Customer Journey Data" of the paper.
2) Using this process model, we generated 1000 traces using the pluging ``Perform a simple simulation of a
(stochastic) Petri net'' in ProM 6.7". The resulting event logs is in {source}
3) The role of this script is to generate 10 customer journeys by concatenating the 1000 traces contained in {source}
as described in the paper. We produce customer journey event logs for various delays. The smaller the delay, the harder it is to retrieve
'''
source = 'source/pm.csv'                                                # 1000 customer journeys produced by Prom
delays = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 1.0, 2.0]   # Delays
n_final_journeys = 10                                                   # Final number of customer journeys
rate = 1                                                                # Rate of the poisson process

# Read original event logs
df = pd.read_csv(source)\
    .drop(['startTime', 'completeTime', 'simulated:logProbability', 'concept:simulated'], axis=1)\
    .reset_index()

# Randomly assign the 1000 cases to 10 journeys
df['journey_id'] = df['case'].map(
    pd.Series(np.random.randint(0, n_final_journeys, size=df['case'].nunique()), index=df['case'].unique()).to_dict()
)
df.sort_values(by=['journey_id', 'index'], inplace=True)

# Choose as the time until next event using a poisson process
df['random_distributed_time'] = simulatePoissonProcess(df.shape[0], rate)

df['new_trace'] = df['case']!=df['case'].shift(1)
df['last_trace'] = df['case']!=df['case'].shift(-1)

# We add a little delay that should help to partition the traces
# The bigger the delays, the easier it is to retrieve the original partition
for delay in delays:
    c = df.copy()
    c['delay'] = 0
    c.loc[c['new_trace']==True, 'delay'] = delay
    c['random_distributed_time_delay'] = c['random_distributed_time']+c['delay']
    c['timestamp'] = c['random_distributed_time_delay'].cumsum()

    c['time_diff'] = c.groupby('journey_id')['timestamp'].shift(-1)-c['timestamp']
    c.drop(['index', 'new_trace', 'random_distributed_time_delay'], axis=1)
    c.to_csv('synthetic_1_delay{}.csv'.format(round(delay,2)))
