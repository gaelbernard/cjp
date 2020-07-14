import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

dataset = {
    'csv_path':'../datasets/XXX.csv',
    'name':'sil',
    'group_by': ' Demandeur',
    'activity_col':' Libelle_intervenant',
    'time_col':" Date_de_fin_de_l'action",
    'guess_col':'N',
    'sep':';',
    'experiment':'real_dataset'
}
df = pd.read_csv('sil.csv', sep=';', nrows=None)

time_col = dataset['time_col']
group_by_col = dataset['group_by']
activity_col = dataset['activity_col']
guess_col = dataset['guess_col']

print (df.shape)

df.sort_values(by=[group_by_col, time_col], inplace=True)
print (df.head().to_string())

for v in [guess_col, group_by_col, time_col, activity_col]:
    df = df[df[v].notna()]

df[time_col] = pd.to_datetime(df[time_col])

# Group by day
#df[group_by_col] = df[group_by_col].astype('str') +'_'+ pd.to_datetime(df[time_col]).dt.date.astype('str')

# Keep only busy CJs
print (df.shape)
nlarg = df.groupby(group_by_col)[guess_col].nunique()
print (nlarg[nlarg>nlarg.quantile(0.99)].to_string())
df = df[df[group_by_col].isin(nlarg[nlarg>nlarg.quantile(0.99)].index)]


#print (nlarg.to_string())
print (df.shape)

# Transform to timestamp
df[time_col] = df[time_col].astype(int)/ 10**9
df['time'] = df[time_col]
print (df.columns)

df['ticket_id'] = df['N'].map({k:y for y,k in pd.Series(df['N'].unique()).items()})
df['customer_id'] = df[' Demandeur'].map({k:y for y,k in pd.Series(df[' Demandeur'].unique()).items()})
df['activity'] = df[' Libelle_intervenant'].map({k:y for y,k in pd.Series(df[' Libelle_intervenant'].unique()).items()})
df['time_diff'] = df.groupby(group_by_col)[time_col].shift(-1)-df[time_col]
df['time_diff_norm'] = np.log(df['time_diff']+1)
df['time_diff_norm'] = MinMaxScaler().fit_transform(df['time_diff_norm'].values.reshape(-1,1))
df = df[['ticket_id', 'customer_id', 'activity', 'time', 'time_diff', 'time_diff_norm']]
df.to_csv('real_transformed.csv')

exit()
df['time_diff_norm'].plot.hist(bins=100)
plt.show()

