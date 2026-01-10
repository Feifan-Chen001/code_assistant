# --- notebook cell 3 ---
import pandas as pd
import numpy as np

# --- notebook cell 6 ---
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = pd.read_csv(url)

iris.head()

# --- notebook cell 8 ---
# 1. sepal_length (in cm)
# 2. sepal_width (in cm)
# 3. petal_length (in cm)
# 4. petal_width (in cm)
# 5. class

iris.columns = ['sepal_length','sepal_width', 'petal_length', 'petal_width', 'class']
iris.head()

# --- notebook cell 10 ---
pd.isnull(iris).sum()
# nice no missing value

# --- notebook cell 12 ---
iris.iloc[10:30,2:3] = np.nan
iris.head(20)

# --- notebook cell 14 ---
iris.petal_length.fillna(1, inplace = True)
iris

# --- notebook cell 16 ---
del iris['class']
iris.head()

# --- notebook cell 18 ---
iris.iloc[0:3 ,:] = np.nan
iris.head()

# --- notebook cell 20 ---
iris = iris.dropna(how='any')
iris.head()

# --- notebook cell 22 ---
iris = iris.reset_index(drop = True)
iris.head()