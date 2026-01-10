# --- notebook cell 3 ---
import pandas as pd

# --- notebook cell 5 ---
raw_data = {"name": ['Bulbasaur', 'Charmander','Squirtle','Caterpie'],
            "evolution": ['Ivysaur','Charmeleon','Wartortle','Metapod'],
            "type": ['grass', 'fire', 'water', 'bug'],
            "hp": [45, 39, 44, 45],
            "pokedex": ['yes', 'no','yes','no']                        
            }

# --- notebook cell 7 ---
pokemon = pd.DataFrame(raw_data)
pokemon.head()

# --- notebook cell 9 ---
pokemon = pokemon[['name', 'type', 'hp', 'evolution','pokedex']]
pokemon

# --- notebook cell 11 ---
pokemon['place'] = ['park','street','lake','forest']
pokemon

# --- notebook cell 13 ---
pokemon.dtypes