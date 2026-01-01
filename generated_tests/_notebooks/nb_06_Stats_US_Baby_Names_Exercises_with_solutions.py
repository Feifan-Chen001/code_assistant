# --- notebook cell 3 ---
import pandas as pd

# --- notebook cell 6 ---
baby_names = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/06_Stats/US_Baby_Names/US_Baby_Names_right.csv')
baby_names.info()

# --- notebook cell 8 ---
baby_names.head(10)

# --- notebook cell 10 ---
# deletes Unnamed: 0
del baby_names['Unnamed: 0']

# deletes Unnamed: 0
del baby_names['Id']

baby_names.head()

# --- notebook cell 12 ---
baby_names.groupby("Year").sum().idxmax()

# --- notebook cell 14 ---
baby_names['Gender'].value_counts()

# --- notebook cell 16 ---
# you don't want to sum the Year column, so you delete it
del baby_names["Year"]

# group the data
names = baby_names.groupby("Name").sum()

# print the first 5 observations
names.head()

# print the size of the dataset
print(names.shape)

# sort it from the biggest value to the smallest one
names.sort_values("Count", ascending = 0).head()

# --- notebook cell 18 ---
# as we have already grouped by the name, all the names are unique already. 
# get the length of names
len(names)

# --- notebook cell 20 ---
names.Count.idxmax()

# OR

# names[names.Count == names.Count.max()]

# --- notebook cell 22 ---
len(names[names.Count == names.Count.min()])

# --- notebook cell 24 ---
names[names.Count == names.Count.median()]

# --- notebook cell 26 ---
names.Count.std()

# --- notebook cell 28 ---
names.describe()