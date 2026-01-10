# --- notebook cell 3 ---
import pandas as pd

# --- notebook cell 6 ---
euro12 = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/02_Filtering_%26_Sorting/Euro12/Euro_2012_stats_TEAM.csv', sep=',')
euro12

# --- notebook cell 8 ---
euro12.Goals

# --- notebook cell 10 ---
euro12.shape[0]

# --- notebook cell 12 ---
euro12.info()

# --- notebook cell 14 ---
# filter only giving the column names

discipline = euro12[['Team', 'Yellow Cards', 'Red Cards']]
discipline

# --- notebook cell 16 ---
discipline.sort_values(['Red Cards', 'Yellow Cards'], ascending = False)

# --- notebook cell 18 ---
round(discipline['Yellow Cards'].mean())

# --- notebook cell 20 ---
euro12[euro12.Goals > 6]

# --- notebook cell 22 ---
euro12[euro12.Team.str.startswith('G')]

# --- notebook cell 24 ---
# use .iloc to slices via the position of the passed integers
# : means all, 0:7 means from 0 to 7

euro12.iloc[: , 0:7]

# --- notebook cell 26 ---
# use negative to exclude the last 3 columns

euro12.iloc[: , :-3]

# --- notebook cell 28 ---
# .loc is another way to slice, using the labels of the columns and indexes

euro12.loc[euro12.Team.isin(['England', 'Italy', 'Russia']), ['Team','Shooting Accuracy']]