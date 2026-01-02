# --- notebook cell 3 ---
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns


# print the graphs in the notebook

# set seaborn style to white
sns.set_style("white")

# --- notebook cell 6 ---
url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/07_Visualization/Tips/tips.csv'
tips = pd.read_csv(url)

tips.head()

# --- notebook cell 8 ---
del tips['Unnamed: 0']

tips.head()

# --- notebook cell 10 ---
# create histogram
ttbill = sns.distplot(tips.total_bill);

# set lables and titles
ttbill.set(xlabel = 'Value', ylabel = 'Frequency', title = "Total Bill")

# take out the right and upper borders
sns.despine()

# --- notebook cell 12 ---
sns.jointplot(x ="total_bill", y ="tip", data = tips)

# --- notebook cell 14 ---
sns.pairplot(tips)

# --- notebook cell 16 ---
sns.stripplot(x = "day", y = "total_bill", data = tips, jitter = True);

# --- notebook cell 18 ---
sns.stripplot(x = "tip", y = "day", hue = "sex", data = tips, jitter = True);

# --- notebook cell 20 ---
sns.boxplot(x = "day", y = "total_bill", hue = "time", data = tips);

# --- notebook cell 22 ---
# better seaborn style
sns.set(style = "ticks")

# creates FacetGrid
g = sns.FacetGrid(tips, col = "time")
g.map(plt.hist, "tip");

# --- notebook cell 24 ---
g = sns.FacetGrid(tips, col = "sex", hue = "smoker")
g.map(plt.scatter, "total_bill", "tip", alpha =.7)

g.add_legend();