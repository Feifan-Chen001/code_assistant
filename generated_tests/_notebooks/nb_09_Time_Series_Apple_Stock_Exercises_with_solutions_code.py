# --- notebook cell 3 ---
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt


# --- notebook cell 6 ---
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/09_Time_Series/Apple_Stock/appl_1980_2014.csv'
apple = pd.read_csv(url)

apple.head()

# --- notebook cell 8 ---
apple.dtypes

# --- notebook cell 10 ---
apple.Date = pd.to_datetime(apple.Date)

apple['Date'].head()

# --- notebook cell 12 ---
apple = apple.set_index('Date')

apple.head()

# --- notebook cell 14 ---
# NO! All are unique
apple.index.is_unique

# --- notebook cell 16 ---
apple.sort_index(ascending = True).head()

# --- notebook cell 18 ---
apple_month = apple.resample('BM').mean()

apple_month.head()

# --- notebook cell 20 ---
(apple.index.max() - apple.index.min()).days

# --- notebook cell 22 ---
apple_months = apple.resample('BM').mean()

len(apple_months.index)

# --- notebook cell 24 ---
# makes the plot and assign it to a variable
appl_open = apple['Adj Close'].plot(title = "Apple Stock")

# changes the size of the graph
fig = appl_open.get_figure()
fig.set_size_inches(13.5, 9)