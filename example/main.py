import pandas as pd
from LSTM.lstmCut import LstmCut

'''
This is a very simple example to describe how customer journey partitioning can be performed using the
3 techniques described in the paper. Let's say we would like to partition the event logs synthetic_1_delay2.0.csv
which contains 10 customer journeys into 1000 distinct event logs
'''
k = 1000

# STEP 1: reading CSV and preprocessing
path = '../datasets/synthetic/synthetic_1_delay2.0.csv'

group_by = 'journey_id' # CSV_COLUMN: Long running cases we would like to partition
activity = 'event'      # CSV_COLUMN: event
time = 'timestamp'      # CSV_COLUMN: timestamp column

# Read CSV
df = pd.read_csv(path, nrows=None, dtype={group_by:str, activity:str})
df.sort_values(by=[group_by, time], inplace=True)

# Calculate time difference until next event
df['time_diff'] = df.groupby('journey_id')[time].shift(-1)-df[time]
df['time_diff'] = df['time_diff'].fillna(df['time_diff'].max())

# METHOD 1: TAP (using only the time to predict the case id)
# We simply insert a cut at the largest time difference
# and use a cumsum to assign a case_id
df['TAP_is_cut'] = False
df.loc[df['time_diff'].nlargest(k).index, 'TAP_is_cut'] = True
df['TAP_discovered_case'] = df['TAP_is_cut'].shift(1).cumsum().fillna(0)

# METHOD 2: LCPAP (using the mean time between pairs of events)
# Same as method 1 but we replace the true time difference
# by the average time difference per pair of events
df['next_activity'] = df[activity].shift(-1)
df['pair'] = df[activity].astype(str) + '_' + df['next_activity'].astype(str)
mapping = df.groupby('pair')['time_diff'].mean()
df['MPTAP'] = df['pair'].map(mapping)
df['MPTAP_is_cut'] = False
df.loc[df['MPTAP'].nlargest(k).index, 'MPTAP_is_cut'] = True
df['MPTAP_discovered_case'] = df['MPTAP_is_cut'].shift(1).cumsum().fillna(0)

# METHOD 3: GCPAP (using a rich set of contextual data to partition the journey)
# Same as method 1 but we first use LSTM to predict the time until next event and use this value
# instead of the true time difference
lstm = LstmCut(df, group_by, 'time_diff', activity, name='output')
lstm.build_model(epoch=3)
df['GCPAP'] = lstm.predict()
df['GCPAP_is_cut'] = False
df.loc[df['GCPAP'].nlargest(k).index, 'GCPAP_is_cut'] = True
df['GCPAP_discovered_case'] = df['GCPAP_is_cut'].shift(1).cumsum().fillna(0)

print (df.head(100).to_string())
