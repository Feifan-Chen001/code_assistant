# --- notebook cell 3 ---
import pandas as pd

# --- notebook cell 6 ---
users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', 
                      sep='|', index_col='user_id')

# --- notebook cell 8 ---
users.head(25)

# --- notebook cell 10 ---
users.tail(10)

# --- notebook cell 12 ---
users.shape[0]

# --- notebook cell 14 ---
users.shape[1]

# --- notebook cell 16 ---
users.columns

# --- notebook cell 18 ---
# "the index" (aka "the labels")
users.index

# --- notebook cell 20 ---
users.dtypes

# --- notebook cell 22 ---
users.occupation

#or

users['occupation']

# --- notebook cell 24 ---
users.occupation.nunique()
#or by using value_counts() which returns the count of unique elements
#users.occupation.value_counts().count()

# --- notebook cell 26 ---
#Because "most" is asked
users.occupation.value_counts().head(1).index[0]

#or
#to have the top 5

# users.occupation.value_counts().head()

# --- notebook cell 28 ---
users.describe() #Notice: by default, only the numeric columns are returned. 

# --- notebook cell 30 ---
users.describe(include = "all") #Notice: By default, only the numeric columns are returned.

# --- notebook cell 32 ---
users.occupation.describe()

# --- notebook cell 34 ---
round(users.age.mean())

# --- notebook cell 36 ---
users.age.value_counts().tail() #7, 10, 11, 66 and 73 years -> only 1 occurrence