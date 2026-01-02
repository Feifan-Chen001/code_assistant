# --- notebook cell 3 ---
import pandas as pd

# --- notebook cell 6 ---
url = 'https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv'
df = pd.read_csv(url)
df.head()

# --- notebook cell 8 ---
# weekly data

# --- notebook cell 10 ---
df = df.set_index('Date')
df.head()

# --- notebook cell 12 ---
df.index
# it is a 'object' type

# --- notebook cell 14 ---
df.index = pd.to_datetime(df.index)
type(df.index)

# --- notebook cell 16 ---
monthly = df.resample('M').sum()
monthly

# --- notebook cell 18 ---
monthly = monthly.dropna()
monthly

# --- notebook cell 20 ---
year = monthly.resample('AS-JAN').sum()
year