# --- notebook cell 3 ---
import numpy as np
import pandas as pd

# --- notebook cell 6 ---
url = "https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv"
crime = pd.read_csv(url)
crime.head()

# --- notebook cell 8 ---
crime.info()

# --- notebook cell 10 ---
# pd.to_datetime(crime)
crime.Year = pd.to_datetime(crime.Year, format='%Y')
crime.info()

# --- notebook cell 12 ---
crime = crime.set_index('Year', drop = True)
crime.head()

# --- notebook cell 14 ---
del crime['Total']
crime.head()

# --- notebook cell 16 ---
# To learn more about .resample (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html)
# To learn more about Offset Aliases (http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases)

# Uses resample to sum each decade
crimes = crime.resample('10AS').sum()

# Uses resample to get the max value only for the "Population" column
population = crime['Population'].resample('10AS').max()

# Updating the "Population" column
crimes['Population'] = population

crimes

# --- notebook cell 18 ---
# apparently the 90s was a pretty dangerous time in the US
crimes.idxmax(0)