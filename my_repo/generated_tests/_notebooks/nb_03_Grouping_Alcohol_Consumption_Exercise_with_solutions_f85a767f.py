# --- notebook cell 3 ---
import pandas as pd

# --- notebook cell 6 ---
drinks = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv',keep_default_na=False)
drinks.head()

# --- notebook cell 8 ---
drinks.groupby('continent').beer_servings.mean()

# --- notebook cell 10 ---
drinks.groupby('continent').wine_servings.describe()

# --- notebook cell 12 ---
drinks.groupby('continent').mean(numeric_only=True)

# --- notebook cell 14 ---
drinks.groupby('continent').median(numeric_only=True)

# --- notebook cell 16 ---
drinks.groupby('continent').spirit_servings.agg(['mean', 'min', 'max'])