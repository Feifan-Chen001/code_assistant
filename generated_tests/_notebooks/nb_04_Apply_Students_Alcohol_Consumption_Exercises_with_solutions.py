# --- notebook cell 3 ---
import pandas as pd
import numpy

# --- notebook cell 6 ---
csv_url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/Students_Alcohol_Consumption/student-mat.csv'
df = pd.read_csv(csv_url)
df.head()

# --- notebook cell 8 ---
stud_alcoh = df.loc[: , "school":"guardian"]
stud_alcoh.head()

# --- notebook cell 10 ---
capitalizer = lambda x: x.capitalize()

# --- notebook cell 12 ---
stud_alcoh['Mjob'].apply(capitalizer)
stud_alcoh['Fjob'].apply(capitalizer)

# --- notebook cell 14 ---
stud_alcoh.tail()

# --- notebook cell 16 ---
stud_alcoh['Mjob'] = stud_alcoh['Mjob'].apply(capitalizer)
stud_alcoh['Fjob'] = stud_alcoh['Fjob'].apply(capitalizer)
stud_alcoh.tail()

# --- notebook cell 18 ---
def majority(x):
    if x > 17:
        return True
    else:
        return False

# --- notebook cell 19 ---
stud_alcoh['legal_drinker'] = stud_alcoh['age'].apply(majority)
stud_alcoh.head()

# --- notebook cell 21 ---
def times10(x):
    if type(x) is int:
        return 10 * x
    return x

# --- notebook cell 22 ---
stud_alcoh.applymap(times10).head(10)