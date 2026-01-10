# --- notebook cell 3 ---
import numpy as np
import pandas as pd

# package to extract data from various Internet sources into a DataFrame
# make sure you have it installed
import pandas_datareader.data as web

# package for dates
import datetime as dt

# --- notebook cell 5 ---
start_dt = dt.datetime(2015, 1, 1, 0, 0)
start_dt

# --- notebook cell 8 ---
df_apple = web.DataReader("AAPL", "av-daily", start=start_dt, api_key="AZOBAQ2SK8AC1MUD")
df_apple

# --- notebook cell 10 ---
df_apple['stock'] = 'AAPL'
df_apple

# --- notebook cell 12 ---
df_tesla = web.DataReader("TSLA", "av-daily", start=start_dt, api_key="AZOBAQ2SK8AC1MUD")
df_tesla['stock'] = "TSLA"

df_ibm = web.DataReader("IBM", "av-daily", start=start_dt, api_key="AZOBAQ2SK8AC1MUD")
df_ibm['stock'] = "IBM"

df_microsoft = web.DataReader("MSFT", "av-daily", start=start_dt, api_key="AZOBAQ2SK8AC1MUD")
df_microsoft['stock'] = "MSFT"

# --- notebook cell 14 ---
df_all = pd.concat([df_apple, df_tesla, df_ibm, df_microsoft])
df_all

# --- notebook cell 16 ---
df_all = df_all.set_index('stock', append=True)
df_all

# --- notebook cell 18 ---
vol = df_all.filter(['volume'])
vol

# --- notebook cell 20 ---
vol_week = vol.rename_axis(index=['dt', 'stock'])
vol_week.reset_index(inplace=True)
vol_week.dt = pd.to_datetime(vol_week.dt)
vol_week['year'] = vol_week['dt'].map(lambda x: x.year)
vol_week['week'] = vol_week['dt'].map(lambda x: x.week)
vol_week.pop('dt')

vol_week.set_index(['year', 'week', 'stock'], inplace=True)
vol_week = vol_week.groupby([pd.Grouper(level='stock'), pd.Grouper(level='year'), pd.Grouper(level='week')]).sum()

vol_week = vol_week.reset_index()
vol_week = vol_week.pivot(index=['year', 'week'], columns='stock')
vol_week.columns.droplevel(0)
vol_week.columns = vol_week.columns.droplevel(0)
vol_week

# --- notebook cell 22 ---
vol_year = vol_week.groupby('year').sum()
vol_year.loc[2015:2015]