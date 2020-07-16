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

# Synthetic dataset
template = {
    'csv_path':'../datasets/synthetic/synthetic_1_delay$$$.csv', # Path of the dataset
    'name':'synthetic_$$$',                                      # Name of the experiment
    'group_by': 'journey_id',                                    # COLUMN Case ID (long-running and complex cases)
    'activity':'event',                                          # COLUMN activity
    'time_diff':'time_diff',                                     # COLUMN time until next event
    'time':'timestamp',                                          # COLUMN timestamp (used for sorting)
    'guess':'case',                                              # COLUMN ground truth (true case id)
}
for delay in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.5, 1.0, 2.0]:
    dataset = copy.deepcopy(template)
    dataset['name'] = dataset['name'].replace('$$$', str(round(delay,2)))
    dataset['csv_path'] = dataset['csv_path'].replace('$$$', str(round(delay,2)))
    datasets.append(dataset)

# Real life dataset
datasets.append(
    {
    'csv_path':'../datasets/real/real_transformed.csv',
    'name':'real_ds',
    'group_by': 'customer_id',
    'activity':'activity',
    'time_diff':'time_diff_norm',
    'time': 'time',
    'guess':'ticket_id',
    }
)

# Timestamp of the experiment
final_t = round(time.time(),0)
final_output = []

for dataset in datasets:
    output = copy.deepcopy(dataset)

    # Create unique ID for the experiment (using the timestamp and the name of the ds)
    t = '{}_{}'.format(round(time.time(),0), dataset['name'])
    os.mkdir('results/{}'.format(t))

    # Read CSV
    dtype = {dataset['guess']:str, dataset['group_by']:str, dataset['activity']:str}
    df = pd.read_csv('{}'.format(dataset['csv_path']), nrows=None, dtype=dtype)
    df.sort_values(by=[dataset['group_by'], dataset['time']], inplace=True)
    df = df[df[dataset['time_diff']].notna()]

    # 'next_guess_col' is the ground truth, i.e., what we try to retrieve
    df['next_guess_col'] = df[dataset['guess']].shift(-1)
    df['new_guess_col'] = (df['next_guess_col']!=df[dataset['guess']])|(df['next_guess_col'].isna())
    df.drop(['next_guess_col'], axis=1, inplace=True)

    # Build LSTM to predict time until next event
    output['window'] = 10   # lookout window size
    output['epoch'] = 20    # Number of epochs
    output['factor'] = 64   # Number of cells in LSTM
    output['noise'] = 0.5   # Dropout in LSTM
    lstm = LstmCut(df, dataset['group_by'], dataset['time_diff'], dataset['activity'], name='results/{}/{}'.format(t, dataset['name'], factor=output['factor']), window=output['window'], noise=output['noise'])

    # GCPAP (Global Context Process Aware Partitioning)
    exec_time = {}
    exec_time['GCPAP'] = lstm.build_model(epoch=output['epoch'])
    df['GCPAP'] = lstm.predict()

    # TAP (Time-aware partitioning)
    # Takes the real time between events
    # We just make sure that the last event of a group by is turn in a np.nan
    # (because the time until next event for the last event does not make sense)
    start = time.time()
    df['TAP'] = df[dataset['time_diff']]
    df.loc[df['split']=='last', 'TAP'] = np.nan
    exec_time['TAP'] = time.time() - start

    # LCPAP (local context process-aware partitioning)
    # The mean pair time aware partitioning
    start = time.time()
    df['next_activity'] = df[dataset['activity']].shift(-1)
    df['pair'] = df[dataset['activity']].astype(str) + '_' + df['next_activity'].astype(str)
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
