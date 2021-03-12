import pandas as pd
df = pd.read_csv('output_df.csv')

def get_n_variant(df, col='partition_id'):
    return df.groupby(col)['activity'].agg(list).value_counts().shape

original_n_tickets = df['activity'].nunique()
print (original_n_tickets)
exit()
df['last_customer'] = df['customer_id'].shift(-1) != df['customer_id']
n_cuts = original_n_tickets - df['last_customer'].sum()
print (df.columns)
print ('ORIGINAL', get_n_variant(df, 'ticket_id'))

for technique in ['GCPAP', 'MPTAP', 'TAP']:
    df['partition'] = df['last_customer'].copy()
    df.loc[df[technique].nlargest(n_cuts).index, 'partition'] = True
    df['partition_id'] = df['partition'].shift(1).cumsum().fillna(0)
    print (technique, get_n_variant(df, 'partition_id'))

exit()

print (df.head(5000).to_string())
exit()