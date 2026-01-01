# --- notebook cell 3 ---
import pandas as pd
import numpy as np

# --- notebook cell 5 ---
s1 = pd.Series(np.random.randint(1, high=5, size=100, dtype='l'))
s2 = pd.Series(np.random.randint(1, high=4, size=100, dtype='l'))
s3 = pd.Series(np.random.randint(10000, high=30001, size=100, dtype='l'))

print(s1, s2, s3)

# --- notebook cell 7 ---
housemkt = pd.concat([s1, s2, s3], axis=1)
housemkt.head()

# --- notebook cell 9 ---
housemkt.rename(columns = {0: 'bedrs', 1: 'bathrs', 2: 'price_sqr_meter'}, inplace=True)
housemkt.head()

# --- notebook cell 11 ---
# join concat the values
bigcolumn = pd.concat([s1, s2, s3], axis=0)

# it is still a Series, so we need to transform it to a DataFrame
bigcolumn = bigcolumn.to_frame()
print(type(bigcolumn))

bigcolumn

# --- notebook cell 13 ---
# no the index are kept but the length of the DataFrame is 300
len(bigcolumn)

# --- notebook cell 15 ---
bigcolumn.reset_index(drop=True, inplace=True)
bigcolumn