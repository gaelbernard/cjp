import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv('/Users/gbernar1/Documents/dev/cutJourneys/experiment/results/in-paper/1592930953.0_real_ds/output_df.csv')
df['true_partitioning'] = df['ticket_id'].shift(-1) != df['ticket_id']
df = df.loc[df['customer_id'].shift(-1) == df['customer_id'],:]
df['bin'] = np.nan
min_v = 0
k = {
    60: '< 1 minute',
    60*60 : '< 1 hour',
    60*60*24 : '< 1 day',
    999999999999999: '> day',
}
for max_v, txt in k.items():
    df.loc[(df['time_diff']>min_v)&(df['time_diff']<=max_v), 'bin'] = txt
    min_v = max_v

print ([x+0.14 for x in range(len(k))])

ax = plt.bar([x-0.14 for x in range(len(k))], df.loc[df['true_partitioning']==False, 'bin'].value_counts(), width=0.25, tick_label=list(k.values()))
plt.bar([x+0.14 for x in range(len(k))], df.loc[df['true_partitioning']==True, 'bin'].value_counts(), width=0.25, tick_label=list(k.values()))
plt.yscale('log')
plt.tight_layout()

plt.legend(["Inter-event", "Inter-journey"], loc="upper right")

plt.ylabel('Count')
plt.xlabel('$\delta$')
plt.savefig('fig.pdf', format='pdf')
print (df.loc[df['true_partitioning']==True, 'time_diff'].median())
print (df.loc[df['true_partitioning']==False, 'time_diff'].median())
exit()
median = df.loc[df['true_partitioning']==True, 'time_diff'].median()
n_true_partitioning = df.loc[df['true_partitioning']==True, 'time_diff'].shape[0]

print ((df.loc[df['true_partitioning']==False, 'time_diff']>median).sum())

print (df.loc[df['true_partitioning']==True, 'time_diff'].median())
print (df.loc[df['true_partitioning']==True, 'time_diff'].shape)
print (df.head().to_string())
output = []
for b in [True, False]:
    if b:
        bv = 'Inter Journey Time Delta (for same customer)'
    else:
        bv = 'Inter Event Time Delta (within same journey)'

    for q in np.arange(0.0,1,0.01):
        output.append({
            'b': bv,
            'Quantile': q,
            'TAP': df.loc[df['true_partitioning']==b, 'TAP'].quantile(q),
            'GCPAP': df.loc[df['true_partitioning']==b, 'GCPAP'].quantile(q),
            'MPTAP': df.loc[df['true_partitioning']==b, 'MPTAP'].quantile(q)
        })


pd.DataFrame(output).set_index('Quantile').groupby('b')['GCPAP'].plot(
    logy=True,
    legend = 'quantile',
    ylabel='Minutes',
)

plt.show()
print ()

#plt.show()
