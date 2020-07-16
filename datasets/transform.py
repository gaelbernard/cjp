import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
The dataset XXX contains confidential information about ticket handling in a service desk of a municipality over a time span of one year. The goal of this file is to anonymzed the dataset and only keep the top $1\%$ of customers that generated most of the tickets throughout the year. By doing so, we focus the experiment on the difficult cases where many tickets are generated within a short period of time. It results in a customer journey event log with the following characteristics: 27 customer journeys, 7.9K tickets, and 100.9K `touchpoints' of 290 types.
'''

dataset = {
    'csv_path':'XXX.csv',
    'name':'XXX',
    'group_by': ' XXX',
    'activity_col':' XXX',
    'time_col':" XXX",
    'guess_col':'XXX',
    'sep':';',
    'experiment':'real_dataset'
}
df = pd.read_csv(dataset['csv_path'], sep=dataset['sep'])

time_col = dataset['time_col']
group_by_col = dataset['group_by']
activity_col = dataset['activity_col']
guess_col = dataset['guess_col']

# PREPROCESSING
# interpret datetime
df[time_col] = pd.to_datetime(df[time_col])

# sort values
df.sort_values(by=[group_by_col, time_col], inplace=True)

# removing lines with nan values
for v in [guess_col, group_by_col, time_col, activity_col]:
    df = df[df[v].notna()]

# TRANSFORM
# Keep only busy CJs
nlarg = df.groupby(group_by_col)[guess_col].nunique()
df = df[df[group_by_col].isin(nlarg[nlarg>nlarg.quantile(0.99)].index)]

# Scale the timestamp
df[time_col] = df[time_col].astype(int)/ 10**9
df['time'] = df[time_col]
df['ticket_id'] = df['N'].map({k:y for y,k in pd.Series(df['N'].unique()).items()})
df['customer_id'] = df[' Demandeur'].map({k:y for y,k in pd.Series(df[' Demandeur'].unique()).items()})
df['activity'] = df[' Libelle_intervenant'].map({k:y for y,k in pd.Series(df[' Libelle_intervenant'].unique()).items()})
df['time_diff'] = df.groupby(group_by_col)[time_col].shift(-1)-df[time_col]
df['time_diff_norm'] = np.log(df['time_diff']+1)
df['time_diff_norm'] = MinMaxScaler().fit_transform(df['time_diff_norm'].values.reshape(-1,1))
df = df[['ticket_id', 'customer_id', 'activity', 'time', 'time_diff', 'time_diff_norm']]
df.to_csv('real_transformed.csv')
