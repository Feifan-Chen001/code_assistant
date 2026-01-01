# --- notebook cell 3 ---
import numpy as np
import pandas as pd

# package to extract data from various Internet sources into a DataFrame
# make sure you have it installed
import pandas_datareader.data as web

# package for dates
import datetime as dt

# --- notebook cell 5 ---
start = dt.datetime(2015, 1, 1)
end = dt.datetime.today()

start

# --- notebook cell 8 ---
df_apple = web.DataReader(name="AAPL",
                          data_source="av-daily",
                          start=start,
                          end=end,
                          api_key="your_alpha_vantage_api_key_goes_here")

df_apple

# --- notebook cell 10 ---
df_apple["stock"] = "AAPL"
df_apple

# --- notebook cell 12 ---
# Tesla
df_tesla = web.DataReader(name="TSLA",
                          data_source="av-daily",
                          start=start,
                          end=end,
                          api_key="your_alpha_vantage_api_key_goes_here")

df_tesla["stock"] = "TSLA"

# IBM
df_ibm = web.DataReader(name="IBM",
                        data_source="av-daily",
                        start=start,
                        end=end,
                        api_key="your_alpha_vantage_api_key_goes_here")

df_ibm["stock"] = "IBM"

# Microsoft
df_microsoft = web.DataReader(name="MSFT",
                              data_source="av-daily",
                              start=start,
                              end=end,
                              api_key="your_alpha_vantage_api_key_goes_here")

df_microsoft["stock"] = "MSFT"

# --- notebook cell 14 ---
frames = [df_apple, df_tesla, df_ibm, df_microsoft]

df = pd.concat(frames)
df

# --- notebook cell 16 ---
df.set_index(keys="stock", append=True, inplace=True)
df

# --- notebook cell 18 ---
vol = df['volume']
vol = pd.DataFrame(vol)
vol

# --- notebook cell 20 ---
date = vol.index.get_level_values(0)
date = pd.DatetimeIndex(date) # ensure that it's a datetimeindex, instead of a regular index

vol['week'] = date.isocalendar().week.values
# .values is necessary to obtain only the week *values*
# otherwise pandas interprets it as a part of an index; this would be a problem as the same week appears multiple times
# (same week number in different years, same week for different stocks)

vol['year'] = date.year

pd.pivot_table(vol, values='volume', index=['year', 'week'],
               columns=['stock'], aggfunc=np.sum)

# --- notebook cell 22 ---
vol_2015 = vol[vol['year'] == 2015]

pd.pivot_table(vol_2015, values='volume', index=['year'],
               columns=['stock'], aggfunc=np.sum)