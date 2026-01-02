# --- notebook cell 4 ---
import pandas as pd
import numpy as np

# --- notebook cell 6 ---
food = pd.read_csv('~/Desktop/en.openfoodfacts.org.products.tsv', sep='\t')

# --- notebook cell 8 ---
food.head()

# --- notebook cell 10 ---
food.shape #will give you both (observations/rows, columns)

# --- notebook cell 11 ---
food.shape[0] #will give you only the observations/rows number

# --- notebook cell 13 ---
print(food.shape) #will give you both (observations/rows, columns)
print(food.shape[1]) #will give you only the columns number

#OR

food.info() #Columns: 163 entries

# --- notebook cell 15 ---
food.columns

# --- notebook cell 17 ---
food.columns[104]

# --- notebook cell 19 ---
food.dtypes['-glucose_100g']

# --- notebook cell 21 ---
food.index

# --- notebook cell 23 ---
food.values[18][7]