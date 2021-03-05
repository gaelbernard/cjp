import numpy as np
import copy
import os
import time
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from LSTM.lstmCut import LstmCut
import pm4py

# DATASETS
datasets = []
base_directory = '/Users/gbernar1/Documents/Dev/0_data/xes-standard-dataset/Leemans_artificial/artificial_event_logs/1 - scalability/generatedLogs/'
journey_col = 'journey_id'
activity_col = 'concept:name'
time_diff_col = 'time_diff'
time_col = 'timestamp'
guess_col = 'case:concept:name'

def read_xes(path, case='case:concept:name', activity_col='concept:name'
):
    log = pm4py.read_xes(path)
    df = pd.DataFrame({
            activity_col: [event[activity_col] for case, trace in enumerate(log) for ts, event in enumerate(trace)],
            case: [case for case, trace in enumerate(log) for ts, event in enumerate(trace)]
        })
    return df

for round_n in list(range(1,11)):
    for treeSeed in list(range(1,11)):
        dataset = {}
        dataset['name'] = 'round{}_treeseed{}'.format(round_n, treeSeed)
        dataset['treeSeed'] = treeSeed
        dataset['xes_path'] = '{}round {} treeSeed {}.xes.gz'.format(base_directory, round_n, treeSeed)
        datasets.append(dataset)

# Timestamp of the experiment
final_t = round(time.time(),0)
final_output = []

for dataset in datasets:
    output = copy.deepcopy(dataset)

    # Create unique ID for the experiment (using the timestamp and the name of the ds)
    t = '{}_{}'.format(round(time.time(),0), dataset['name'])
    os.mkdir('results-March03/{}'.format(t))

    # Read CSV
    dtype = {guess_col:str, journey_col:str, activity_col:str}
    df = read_xes('{}'.format(dataset['xes_path']))
    df['journey_id'] = 1
    assignement = np.random.randint(low=1,high=10,size=df[guess_col].nunique())
    df['journey_id'] = df[guess_col].map(pd.Series(assignement, index=df[guess_col].unique()).to_dict())
    df[time_diff_col] = np.random.rand(df.shape[0])

    # 'next_guess_col' is the ground truth, i.e., what we try to retrieve
    df['next_guess_col'] = df[guess_col].shift(-1)
    df['new_guess_col'] = (df['next_guess_col']!=df[guess_col])|(df['next_guess_col'].isna())
    df.drop(['next_guess_col'], axis=1, inplace=True)




    # Build LSTM to predict time until next event
    output['window'] = 10   # lookout window size
    output['epoch'] = 20    # Number of epochs
    output['factor'] = 64   # Number of cells in LSTM
    output['noise'] = 0.5   # Dropout in LSTM

    lstm = LstmCut(df, journey_col, time_diff_col, activity_col, name='results-March03/{}/{}'.format(t, dataset['name'], factor=output['factor']), window=output['window'], noise=output['noise'])

    # GCPAP (Global Context Process Aware Partitioning)
    exec_time = {}
    exec_time['GCPAP'] = lstm.build_model(epoch=output['epoch'])
    df['GCPAP'] = lstm.predict()

    # TAP (Time-aware partitioning)
    # Takes the real time between events
    # We just make sure that the last event of a group by is turn in a np.nan
    # (because the time until next event for the last event does not make sense)
    start = time.time()
    df['TAP'] = df[time_diff_col]
    df.loc[df['split']=='last', 'TAP'] = np.nan
    exec_time['TAP'] = time.time() - start

    # LCPAP (local context process-aware partitioning)
    # The mean pair time aware partitioning
    start = time.time()
    df['next_activity'] = df[activity_col].shift(-1)
    df['pair'] = df[activity_col].astype(str) + '_' + df['next_activity'].astype(str)
    mapping = df.groupby('pair')['TAP'].mean()
    df['LCPAP'] = df['pair'].map(mapping)
    exec_time['LCPAP'] = time.time() - start



    for c in ['GCPAP', 'TAP', 'LCPAP']:
        o = copy.deepcopy(output)
        o['exec_time'] = exec_time[c]
        o['type'] = c
        final_output.append(o)

    # Export results
    frame = pd.DataFrame(final_output)
    frame.to_csv('results-March03/{}.csv'.format(final_t))
