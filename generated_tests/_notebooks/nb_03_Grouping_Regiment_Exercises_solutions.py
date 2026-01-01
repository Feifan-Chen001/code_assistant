# --- notebook cell 3 ---
import pandas as pd

# --- notebook cell 5 ---
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'], 
        'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'], 
        'name': ['Miller', 'Jacobson', 'Ali', 'Milner', 'Cooze', 'Jacon', 'Ryaner', 'Sone', 'Sloan', 'Piger', 'Riani', 'Ali'], 
        'preTestScore': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
        'postTestScore': [25, 94, 57, 62, 70, 25, 94, 57, 62, 70, 62, 70]}

# --- notebook cell 7 ---
regiment = pd.DataFrame(raw_data, columns = raw_data.keys())
regiment

# --- notebook cell 9 ---
regiment[regiment['regiment'] == 'Nighthawks'].groupby('regiment').mean()

# --- notebook cell 11 ---
regiment.groupby('company').describe()

# --- notebook cell 13 ---
regiment.groupby('company').preTestScore.mean()

# --- notebook cell 15 ---
regiment.groupby(['regiment', 'company']).preTestScore.mean()

# --- notebook cell 17 ---
regiment.groupby(['regiment', 'company']).preTestScore.mean().unstack()

# --- notebook cell 19 ---
regiment.groupby(['regiment', 'company']).mean()

# --- notebook cell 21 ---
regiment.groupby(['company', 'regiment']).size()

# --- notebook cell 23 ---
# Group the dataframe by regiment, and for each regiment,
for name, group in regiment.groupby('regiment'):
    # print the name of the regiment
    print(name)
    # print the data of that regiment
    print(group)