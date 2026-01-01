# --- notebook cell 3 ---
import pandas as pd
import numpy as np

# --- notebook cell 6 ---
cars1 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars1.csv")
cars2 = pd.read_csv("https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/05_Merge/Auto_MPG/cars2.csv")

print(cars1.head())
print(cars2.head())

# --- notebook cell 8 ---
cars1 = cars1.loc[:, "mpg":"car"]
cars1.head()

# --- notebook cell 10 ---
print(cars1.shape)
print(cars2.shape)

# --- notebook cell 12 ---
cars = cars1.append(cars2)
cars

# --- notebook cell 14 ---
nr_owners = np.random.randint(15000, high=73001, size=398, dtype='l')
nr_owners

# --- notebook cell 16 ---
cars['owners'] = nr_owners
cars.tail()