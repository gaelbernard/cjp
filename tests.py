import pandas as pd
import numpy as np
from random import expovariate
import pm4py

def read_xes(path):
    log = pm4py.read_xes(path)
    from pm4py.objects.conversion.log import converter as log_converter
    return log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)


# /Users/gbernar1/Documents/Dev/0_data/Process Mining/BPI_Challenge_2012.xes NO
#df = pd.read_csv('/Users/gbernar1/Documents/Dev/0_data/xes-standard-dataset/Hospital Billing - Event Log.csv', nrows=100000000)
#df.sort_values(['Resource','Complete Timestamp'], inplace=True)
#print (df.head(1000).to_string())

path = "/Users/gbernar1/Documents/Dev/0_data/xes-standard-dataset/bpi_2020/PermitLog.xes.gz"
df = read_xes(path)
print (df.head().to_string())
df.sort_values(['case:OrganizationalEntity','time:timestamp'], inplace=True)
print (df[['case:OrganizationalEntity','time:timestamp','case:travel permit number','concept:name']].head(10000).to_string())