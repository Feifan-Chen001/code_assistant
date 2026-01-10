# --- notebook cell 3 ---
import pandas as pd

# --- notebook cell 6 ---
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'

chipo = pd.read_csv(url, sep = '\t')

# --- notebook cell 8 ---
# the item price column is actullay the price of the product multiplied by the quantity
chipo.loc[(chipo["choice_description"] == '[Diet Coke]') & (chipo["item_name"] == "Canned Soda")]

# --- notebook cell 9 ---
# adding a new column representing the price of each single product in float
chipo["item_price"] = chipo["item_price"].str.replace("$", "", regex=False).astype(float)
chipo["product_price"] = chipo["item_price"] / chipo["quantity"]
chipo

# --- notebook cell 10 ---
#checking everything is correct
chipo.loc[(chipo["choice_description"] == '[Diet Coke]') & (chipo["item_name"] == "Canned Soda")]

# --- notebook cell 11 ---
# removing duplicated products
filtered_chipo=chipo.drop_duplicates(['item_name','choice_description'])

# --- notebook cell 12 ---
# filtering products that costs more than $10
filtered_chipo = filtered_chipo.loc[ filtered_chipo["product_price"]>10.0 , ["item_name","choice_description","product_price"] ].reset_index(drop=True)

# --- notebook cell 13 ---
print(f"the number of products that cost more than $10.00 is {filtered_chipo.shape[0]}")

# --- notebook cell 15 ---
filtered_chipo[["item_name","choice_description","product_price"]]

# --- notebook cell 17 ---
chipo.item_name.sort_values()

# OR

chipo.sort_values(by = "item_name")

# --- notebook cell 19 ---
chipo.loc[chipo["item_price"].idxmax()]["quantity"]

# --- notebook cell 21 ---
chipo[chipo["item_name"]=="Veggie Salad Bowl"]["quantity"].sum()

# --- notebook cell 23 ---
chipo[ ( chipo["item_name"]=="Canned Soda" ) & ( chipo["quantity"]>1 )].shape[0]