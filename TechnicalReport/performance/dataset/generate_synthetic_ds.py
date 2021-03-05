import pandas as pd
import numpy as np
from random import expovariate
import pm4py

def simulatePoissonProcess(size, rate):
    return [expovariate(1/rate) for _ in range(size)]

'''
1) We leverage the event logs from https://data.4tu.nl/articles/dataset/A_collection_of_artificial_event_logs_to_test_process_discovery_and_conformance_checking_techniques/12704777
2) We focus on the xes in the folder "1 - scalability", "round 5" 
3) The role of this script is to generate 10 customer journeys by concatenating the 1024 traces
as described in the paper. We produce customer journey event logs for various delays. The smaller the delay, the harder it is to retrieve
'''

def read_xes(path, case='case:concept:name', activity_col='concept:name'
):
    log = pm4py.read_xes(path)
    df = pd.DataFrame({
            activity_col: [event[activity_col] for case, trace in enumerate(log) for ts, event in enumerate(trace)],
            case: [case for case, trace in enumerate(log) for ts, event in enumerate(trace)]
        })
    return df

# Params
base_folder = "original_synthetic_ds/1 - scalability/generatedLogs/"
output_folder = "journey_synthetic/"
round_n = 5
treeSeeds = list(range(1,11))
delays = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 1.0, 2.0, 5.0]   # Delays
n_final_journeys = 10                                              # Final number of customer journeys
rate = 1                                                           # Rate of the poisson process
case_col = 'case:concept:name'
activity_col = 'concept:name'

for treeSeed in treeSeeds:
    df = read_xes('{}/round {} treeSeed {}.xes.gz'.format(base_folder, round_n, treeSeed)).reset_index()

    # Randomly assign the 1000 cases to 10 journeys
    df['journey_id'] = df[case_col].map(
        pd.Series(np.random.randint(0, n_final_journeys, size=df[case_col].nunique()), index=df[case_col].unique()).to_dict()
    )
    df.sort_values(by=['journey_id', 'index'], inplace=True)

    # Choose as the time until next event using a poisson process
    df['random_distributed_time'] = simulatePoissonProcess(df.shape[0], rate)

    df['new_trace'] = df[case_col]!=df[case_col].shift(1)
    df['last_trace'] = df[case_col]!=df[case_col].shift(-1)

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
        c.to_csv('{}round{}_treeseed{}_delay{}.csv'.format(output_folder, round_n, treeSeed, round(delay, 2)))
