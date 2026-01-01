# --- notebook cell 3 ---
import pandas as pd
import numpy as np

# --- notebook cell 6 ---
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
wine = pd.read_csv(url)

wine.head()

# --- notebook cell 8 ---
wine = wine.drop(wine.columns[[0,3,6,8,11,12,13]], axis = 1)

wine.head()

# --- notebook cell 10 ---
wine.columns = ['alcohol', 'malic_acid', 'alcalinity_of_ash', 'magnesium', 'flavanoids', 'proanthocyanins', 'hue']
wine.head()

# --- notebook cell 12 ---
wine.iloc[0:3, 0] = np.nan
wine.head()

# --- notebook cell 14 ---
wine.iloc[2:4, 3] = np.nan
wine.head()

# --- notebook cell 16 ---
wine.alcohol.fillna(10, inplace = True)

wine.magnesium.fillna(100, inplace = True)

wine.head()

# --- notebook cell 18 ---
wine.isnull().sum()

# --- notebook cell 20 ---
random = np.random.randint(10, size = 10)
random

# --- notebook cell 22 ---
wine.alcohol[random] = np.nan
wine.head(10)

# --- notebook cell 24 ---
wine.isnull().sum()

# --- notebook cell 26 ---
wine = wine.dropna(axis = 0, how = "any")
wine.head()

# --- notebook cell 28 ---
mask = wine.alcohol.notnull()
mask

# --- notebook cell 29 ---
wine.alcohol[mask]

# --- notebook cell 31 ---
wine = wine.reset_index(drop = True)
wine.head()