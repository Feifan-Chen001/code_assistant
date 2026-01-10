# --- notebook cell 3 ---
import pandas as pd
import numpy as np

# --- notebook cell 6 ---
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv'
    
chipo = pd.read_csv(url, sep = '\t')

# --- notebook cell 8 ---
chipo.head(10)

# --- notebook cell 10 ---
# Solution 1

chipo.shape[0]  # entries <= 4622 observations

# --- notebook cell 11 ---
# Solution 2

chipo.info() # entries <= 4622 observations

# --- notebook cell 13 ---
chipo.shape[1]

# --- notebook cell 15 ---
chipo.columns

# --- notebook cell 17 ---
chipo.index

# --- notebook cell 19 ---
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)

# --- notebook cell 21 ---
c = chipo.groupby('item_name')
c = c.sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)

# --- notebook cell 23 ---
c = chipo.groupby('choice_description').sum()
c = c.sort_values(['quantity'], ascending=False)
c.head(1)
# Diet Coke 159

# --- notebook cell 25 ---
total_items_orders = chipo.quantity.sum()
total_items_orders

# --- notebook cell 28 ---
chipo.item_price.dtype

# --- notebook cell 30 ---
dollarizer = lambda x: float(x[1:-1])
chipo.item_price = chipo.item_price.apply(dollarizer)

# --- notebook cell 32 ---
chipo.item_price.dtype

# --- notebook cell 34 ---
revenue = (chipo['quantity']* chipo['item_price']).sum()

print('Revenue was: $' + str(np.round(revenue,2)))

# --- notebook cell 36 ---
orders = chipo.order_id.value_counts().count()
orders

# --- notebook cell 38 ---
# Solution 1

chipo['revenue'] = chipo['quantity'] * chipo['item_price']
order_grouped = chipo.groupby(by=['order_id']).sum()
order_grouped.mean()['revenue']

# --- notebook cell 39 ---
# Solution 2

chipo.groupby('order_id')['revenue'].sum().mean()

# --- notebook cell 41 ---
chipo.item_name.value_counts().count()