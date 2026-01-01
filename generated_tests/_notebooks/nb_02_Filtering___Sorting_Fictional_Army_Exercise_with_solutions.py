# --- notebook cell 3 ---
import pandas as pd

# --- notebook cell 5 ---
# Create an example dataframe about a fictional army
raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
            'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
            'deaths': [523, 52, 25, 616, 43, 234, 523, 62, 62, 73, 37, 35],
            'battles': [5, 42, 2, 2, 4, 7, 8, 3, 4, 7, 8, 9],
            'size': [1045, 957, 1099, 1400, 1592, 1006, 987, 849, 973, 1005, 1099, 1523],
            'veterans': [1, 5, 62, 26, 73, 37, 949, 48, 48, 435, 63, 345],
            'readiness': [1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 3],
            'armored': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            'deserters': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
            'origin': ['Arizona', 'California', 'Texas', 'Florida', 'Maine', 'Iowa', 'Alaska', 'Washington', 'Oregon', 'Wyoming', 'Louisana', 'Georgia']}

# --- notebook cell 7 ---
army = pd.DataFrame(data=raw_data)
army

# --- notebook cell 9 ---
army.set_index('origin', inplace=True)

# --- notebook cell 11 ---
army.veterans

# --- notebook cell 13 ---
army[["veterans", "deaths"]]

# --- notebook cell 15 ---
army.columns

# --- notebook cell 17 ---
army.loc[["Maine", "Alaska"], ["deaths", "size", "deserters"]]

# --- notebook cell 19 ---
army.iloc[2:7, 2:6]

# --- notebook cell 21 ---
army.iloc[4:, :]

# --- notebook cell 23 ---
army.iloc[:4, :]

# --- notebook cell 25 ---
army.iloc[:, 2:7]

# --- notebook cell 27 ---
army[army["deaths"] > 50]

# --- notebook cell 29 ---
army[(army["deaths"] > 500) | (army["deaths"] < 50)]

# --- notebook cell 31 ---
army[army["regiment"] != "Dragoons"]

# --- notebook cell 33 ---
army.loc[["Texas", "Arizona"], :]

# --- notebook cell 35 ---
army.loc[["Arizona"]].iloc[:, 2]

# --- notebook cell 37 ---
army.loc[:, ["deaths"]].iloc[2]