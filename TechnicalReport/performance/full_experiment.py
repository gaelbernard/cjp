import numpy as np
import copy
import os
import time
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt
from LSTM.lstmCut import LstmCut

# DATASETS
datasets = []
base_directory = 'dataset/journey_synthetic/'
journey_col = 'journey_id'
activity_col = 'concept:name'
time_diff_col = 'time_diff'
time_col = 'timestamp'
guess_col = 'case:concept:name'
round_n = 5
delays = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 1.0, 2.0, 5.0]
treeSeeds = list(range(1,2))

for delay in delays:
    for treeSeed in treeSeeds:
        dataset = {}
        dataset['name'] = 'round{}_treeseed{}_delay{}'.format(round_n, treeSeed, round(delay,2))
        dataset['treeSeed'] = treeSeed
        dataset['delay'] = delay
        dataset['csv_path'] = '{}{}.csv'.format(base_directory, dataset['name'])
        datasets.append(dataset)

# Timestamp of the experiment
final_t = round(time.time(),0)
final_output = []

for dataset in datasets:
    output = copy.deepcopy(dataset)

    # Create unique ID for the experiment (using the timestamp and the name of the ds)
    t = '{}_{}'.format(round(time.time(),0), dataset['name'])
    os.mkdir('results/{}'.format(t))

    # Read CSV
    dtype = {guess_col:str, journey_col:str, activity_col:str}
    df = pd.read_csv('{}'.format(dataset['csv_path']), nrows=None, dtype=dtype)
    df.sort_values(by=[journey_col, time_col], inplace=True)
    df = df[df[time_diff_col].notna()]

    print (df.head().to_string())

    # 'next_guess_col' is the ground truth, i.e., what we try to retrieve
    df['next_guess_col'] = df[guess_col].shift(-1)
    df['new_guess_col'] = (df['next_guess_col']!=df[guess_col])|(df['next_guess_col'].isna())
    df.drop(['next_guess_col'], axis=1, inplace=True)

    # Build LSTM to predict time until next event
    output['window'] = 10   # lookout window size
    output['epoch'] = 20    # Number of epochs
    output['factor'] = 64   # Number of cells in LSTM
    output['noise'] = 0.5   # Dropout in LSTM
    lstm = LstmCut(df, journey_col, time_diff_col, activity_col, name='results/{}/{}'.format(t, dataset['name'], factor=output['factor']), window=output['window'], noise=output['noise'])

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
    df['MPTAP'] = df['pair'].map(mapping)
    exec_time['MPTAP'] = time.time() - start

    # Save results
    df.drop(['next_activity', 'pair'], axis=1, inplace=True)
    df.to_csv('results/{}/output_df.csv'.format(t))

    # Create ROC curve
    r = df.loc[df['split'] != 'last', :]
    fpr = {}
    tpr = {}
    tresholds = {}
    auc = {}
    for c in ['GCPAP', 'TAP', 'MPTAP']:
        fpr[c], tpr[c], tresholds[c] = roc_curve(r['new_guess_col'], r[c])
        auc[c] = roc_auc_score(r['new_guess_col'], r[c])
        plt.plot(fpr[c], tpr[c], label=c)
        o = copy.deepcopy(output)
        o['exec_time'] = exec_time[c]
        o['type'] = c
        o['auc'] = auc[c]
        final_output.append(o)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.savefig('results/{}/roc.eps'.format(t), format='eps')
    plt.close()

    # Export results
    frame = pd.DataFrame(final_output)
    frame.to_csv('results/{}.csv'.format(final_t))
