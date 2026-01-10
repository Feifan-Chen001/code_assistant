# --- notebook cell 4 ---
from __future__ import division, print_function, unicode_literals

# --- notebook cell 6 ---
import pandas as pd

# --- notebook cell 9 ---
s = pd.Series([2,-1,3,5])
s

# --- notebook cell 11 ---
import numpy as np
np.exp(s)

# --- notebook cell 13 ---
s + [1000,2000,3000,4000]

# --- notebook cell 15 ---
s + 1000

# --- notebook cell 17 ---
s < 0

# --- notebook cell 19 ---
s2 = pd.Series([68, 83, 112, 68], index=["alice", "bob", "charles", "darwin"])
s2

# --- notebook cell 21 ---
s2["bob"]

# --- notebook cell 23 ---
s2[1]

# --- notebook cell 25 ---
s2.loc["bob"]

# --- notebook cell 26 ---
s2.iloc[1]

# --- notebook cell 28 ---
s2.iloc[1:3]

# --- notebook cell 30 ---
surprise = pd.Series([1000, 1001, 1002, 1003])
surprise

# --- notebook cell 31 ---
surprise_slice = surprise[2:]
surprise_slice

# --- notebook cell 33 ---
try:
    surprise_slice[0]
except KeyError as e:
    print("Key error:", e)

# --- notebook cell 35 ---
surprise_slice.iloc[0]

# --- notebook cell 37 ---
weights = {"alice": 68, "bob": 83, "colin": 86, "darwin": 68}
s3 = pd.Series(weights)
s3

# --- notebook cell 39 ---
s4 = pd.Series(weights, index = ["colin", "alice"])
s4

# --- notebook cell 41 ---
print(s2.keys())
print(s3.keys())

s2 + s3

# --- notebook cell 43 ---
s5 = pd.Series([1000,1000,1000,1000])
print("s2 =", s2.values)
print("s5 =", s5.values)

s2 + s5

# --- notebook cell 46 ---
meaning = pd.Series(42, ["life", "universe", "everything"])
meaning

# --- notebook cell 48 ---
s6 = pd.Series([83, 68], index=["bob", "alice"], name="weights")
s6

# --- notebook cell 50 ---
import matplotlib.pyplot as plt
temperatures = [4.4,5.1,6.1,6.2,6.1,6.1,5.7,5.2,4.7,4.1,3.9,3.5]
s7 = pd.Series(temperatures, name="Temperature")
s7.plot()
plt.show()

# --- notebook cell 53 ---
dates = pd.date_range('2016/10/29 5:30pm', periods=12, freq='H')
dates

# --- notebook cell 55 ---
temp_series = pd.Series(temperatures, dates)
temp_series

# --- notebook cell 57 ---
temp_series.plot(kind="bar")

plt.grid(True)
plt.show()

# --- notebook cell 59 ---
temp_series_freq_2H = temp_series.resample("2H")
temp_series_freq_2H

# --- notebook cell 61 ---
temp_series_freq_2H = temp_series_freq_2H.mean()

# --- notebook cell 63 ---
temp_series_freq_2H.plot(kind="bar")
plt.show()

# --- notebook cell 65 ---
temp_series_freq_2H = temp_series.resample("2H").min()
temp_series_freq_2H

# --- notebook cell 67 ---
temp_series_freq_2H = temp_series.resample("2H").apply(np.min)
temp_series_freq_2H

# --- notebook cell 69 ---
temp_series_freq_15min = temp_series.resample("15Min").mean()
temp_series_freq_15min.head(n=10) # `head` displays the top n values

# --- notebook cell 71 ---
temp_series_freq_15min = temp_series.resample("15Min").interpolate(method="cubic")
temp_series_freq_15min.head(n=10)

# --- notebook cell 72 ---
temp_series.plot(label="Period: 1 hour")
temp_series_freq_15min.plot(label="Period: 15 minutes")
plt.legend()
plt.show()

# --- notebook cell 74 ---
temp_series_ny = temp_series.tz_localize("America/New_York")
temp_series_ny

# --- notebook cell 76 ---
temp_series_paris = temp_series_ny.tz_convert("Europe/Paris")
temp_series_paris

# --- notebook cell 78 ---
temp_series_paris_naive = temp_series_paris.tz_localize(None)
temp_series_paris_naive

# --- notebook cell 80 ---
try:
    temp_series_paris_naive.tz_localize("Europe/Paris")
except Exception as e:
    print(type(e))
    print(e)

# --- notebook cell 82 ---
temp_series_paris_naive.tz_localize("Europe/Paris", ambiguous="infer")

# --- notebook cell 84 ---
quarters = pd.period_range('2016Q1', periods=8, freq='Q')
quarters

# --- notebook cell 86 ---
quarters + 3

# --- notebook cell 88 ---
quarters.asfreq("M")

# --- notebook cell 90 ---
quarters.asfreq("M", how="start")

# --- notebook cell 92 ---
quarters.asfreq("A")

# --- notebook cell 94 ---
quarterly_revenue = pd.Series([300, 320, 290, 390, 320, 360, 310, 410], index = quarters)
quarterly_revenue

# --- notebook cell 95 ---
quarterly_revenue.plot(kind="line")
plt.show()

# --- notebook cell 97 ---
last_hours = quarterly_revenue.to_timestamp(how="end", freq="H")
last_hours

# --- notebook cell 99 ---
last_hours.to_period()

# --- notebook cell 101 ---
months_2016 = pd.period_range("2016", periods=12, freq="M")
one_day_after_last_days = months_2016.asfreq("D") + 1
last_bdays = one_day_after_last_days.to_timestamp() - pd.tseries.offsets.BDay()
last_bdays.to_period("H") + 9

# --- notebook cell 103 ---
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)
people

# --- notebook cell 106 ---
people["birthyear"]

# --- notebook cell 108 ---
people[["birthyear", "hobby"]]

# --- notebook cell 110 ---
d2 = pd.DataFrame(
        people_dict,
        columns=["birthyear", "weight", "height"],
        index=["bob", "alice", "eugene"]
     )
d2

# --- notebook cell 112 ---
values = [
            [1985, np.nan, "Biking",   68],
            [1984, 3,      "Dancing",  83],
            [1992, 0,      np.nan,    112]
         ]
d3 = pd.DataFrame(
        values,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )
d3

# --- notebook cell 114 ---
masked_array = np.ma.asarray(values, dtype=np.object)
masked_array[(0, 2), (1, 2)] = np.ma.masked
d3 = pd.DataFrame(
        masked_array,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )
d3

# --- notebook cell 116 ---
d4 = pd.DataFrame(
         d3,
         columns=["hobby", "children"],
         index=["alice", "bob"]
     )
d4

# --- notebook cell 118 ---
people = pd.DataFrame({
    "birthyear": {"alice":1985, "bob": 1984, "charles": 1992},
    "hobby": {"alice":"Biking", "bob": "Dancing"},
    "weight": {"alice":68, "bob": 83, "charles": 112},
    "children": {"bob": 3, "charles": 0}
})
people

# --- notebook cell 120 ---
d5 = pd.DataFrame(
  {
    ("public", "birthyear"):
        {("Paris","alice"):1985, ("Paris","bob"): 1984, ("London","charles"): 1992},
    ("public", "hobby"):
        {("Paris","alice"):"Biking", ("Paris","bob"): "Dancing"},
    ("private", "weight"):
        {("Paris","alice"):68, ("Paris","bob"): 83, ("London","charles"): 112},
    ("private", "children"):
        {("Paris", "alice"):np.nan, ("Paris","bob"): 3, ("London","charles"): 0}
  }
)
d5

# --- notebook cell 122 ---
d5["public"]

# --- notebook cell 123 ---
d5["public", "hobby"]  # Same result as d5["public"]["hobby"]

# --- notebook cell 125 ---
d5

# --- notebook cell 127 ---
d5.columns = d5.columns.droplevel(level = 0)
d5

# --- notebook cell 129 ---
d6 = d5.T
d6

# --- notebook cell 131 ---
d7 = d6.stack()
d7

# --- notebook cell 133 ---
d8 = d7.unstack()
d8

# --- notebook cell 135 ---
d9 = d8.unstack()
d9

# --- notebook cell 137 ---
d10 = d9.unstack(level = (0,1))
d10

# --- notebook cell 140 ---
people

# --- notebook cell 142 ---
people.loc["charles"]

# --- notebook cell 144 ---
people.iloc[2]

# --- notebook cell 146 ---
people.iloc[1:3]

# --- notebook cell 148 ---
people[np.array([True, False, True])]

# --- notebook cell 150 ---
people[people["birthyear"] < 1990]

# --- notebook cell 152 ---
people

# --- notebook cell 153 ---
people["age"] = 2018 - people["birthyear"]  # adds a new column "age"
people["over 30"] = people["age"] > 30      # adds another column "over 30"
birthyears = people.pop("birthyear")
del people["children"]

people

# --- notebook cell 154 ---
birthyears

# --- notebook cell 156 ---
people["pets"] = pd.Series({"bob": 0, "charles": 5, "eugene":1})  # alice is missing, eugene is ignored
people

# --- notebook cell 158 ---
people.insert(1, "height", [172, 181, 185])
people

# --- notebook cell 160 ---
people.assign(
    body_mass_index = people["weight"] / (people["height"] / 100) ** 2,
    has_pets = people["pets"] > 0
)

# --- notebook cell 162 ---
try:
    people.assign(
        body_mass_index = people["weight"] / (people["height"] / 100) ** 2,
        overweight = people["body_mass_index"] > 25
    )
except KeyError as e:
    print("Key error:", e)

# --- notebook cell 164 ---
d6 = people.assign(body_mass_index = people["weight"] / (people["height"] / 100) ** 2)
d6.assign(overweight = d6["body_mass_index"] > 25)

# --- notebook cell 166 ---
try:
    (people
         .assign(body_mass_index = people["weight"] / (people["height"] / 100) ** 2)
         .assign(overweight = people["body_mass_index"] > 25)
    )
except KeyError as e:
    print("Key error:", e)

# --- notebook cell 168 ---
(people
     .assign(body_mass_index = lambda df: df["weight"] / (df["height"] / 100) ** 2)
     .assign(overweight = lambda df: df["body_mass_index"] > 25)
)

# --- notebook cell 171 ---
people.eval("weight / (height/100) ** 2 > 25")

# --- notebook cell 173 ---
people.eval("body_mass_index = weight / (height/100) ** 2", inplace=True)
people

# --- notebook cell 175 ---
overweight_threshold = 30
people.eval("overweight = body_mass_index > @overweight_threshold", inplace=True)
people

# --- notebook cell 177 ---
people.query("age > 30 and pets == 0")

# --- notebook cell 179 ---
people.sort_index(ascending=False)

# --- notebook cell 181 ---
people.sort_index(axis=1, inplace=True)
people

# --- notebook cell 183 ---
people.sort_values(by="age", inplace=True)
people

# --- notebook cell 185 ---
people.plot(kind = "line", x = "body_mass_index", y = ["height", "weight"])
plt.show()

# --- notebook cell 187 ---
people.plot(kind = "scatter", x = "height", y = "weight", s=[40, 120, 200])
plt.show()

# --- notebook cell 190 ---
grades_array = np.array([[8,8,9],[10,9,9],[4, 8, 2], [9, 10, 10]])
grades = pd.DataFrame(grades_array, columns=["sep", "oct", "nov"], index=["alice","bob","charles","darwin"])
grades

# --- notebook cell 192 ---
np.sqrt(grades)

# --- notebook cell 194 ---
grades + 1

# --- notebook cell 196 ---
grades >= 5

# --- notebook cell 198 ---
grades.mean()

# --- notebook cell 200 ---
(grades > 5).all()

# --- notebook cell 202 ---
(grades > 5).all(axis = 1)

# --- notebook cell 204 ---
(grades == 10).any(axis = 1)

# --- notebook cell 206 ---
grades - grades.mean()  # equivalent to: grades - [7.75, 8.75, 7.50]

# --- notebook cell 208 ---
pd.DataFrame([[7.75, 8.75, 7.50]]*4, index=grades.index, columns=grades.columns)

# --- notebook cell 210 ---
grades - grades.values.mean() # subtracts the global mean (8.00) from all grades

# --- notebook cell 212 ---
bonus_array = np.array([[0,np.nan,2],[np.nan,1,0],[0, 1, 0], [3, 3, 0]])
bonus_points = pd.DataFrame(bonus_array, columns=["oct", "nov", "dec"], index=["bob","colin", "darwin", "charles"])
bonus_points

# --- notebook cell 213 ---
grades + bonus_points

# --- notebook cell 215 ---
(grades + bonus_points).fillna(0)

# --- notebook cell 217 ---
fixed_bonus_points = bonus_points.fillna(0)
fixed_bonus_points.insert(0, "sep", 0)
fixed_bonus_points.loc["alice"] = 0
grades + fixed_bonus_points

# --- notebook cell 219 ---
bonus_points

# --- notebook cell 221 ---
bonus_points.interpolate(axis=1)

# --- notebook cell 223 ---
better_bonus_points = bonus_points.copy()
better_bonus_points.insert(0, "sep", 0)
better_bonus_points.loc["alice"] = 0
better_bonus_points = better_bonus_points.interpolate(axis=1)
better_bonus_points

# --- notebook cell 225 ---
grades + better_bonus_points

# --- notebook cell 227 ---
grades["dec"] = np.nan
final_grades = grades + better_bonus_points
final_grades

# --- notebook cell 229 ---
final_grades_clean = final_grades.dropna(how="all")
final_grades_clean

# --- notebook cell 231 ---
final_grades_clean = final_grades_clean.dropna(axis=1, how="all")
final_grades_clean

# --- notebook cell 233 ---
final_grades["hobby"] = ["Biking", "Dancing", np.nan, "Dancing", "Biking"]
final_grades

# --- notebook cell 235 ---
grouped_grades = final_grades.groupby("hobby")
grouped_grades

# --- notebook cell 237 ---
grouped_grades.mean()

# --- notebook cell 240 ---
bonus_points

# --- notebook cell 241 ---
more_grades = final_grades_clean.stack().reset_index()
more_grades.columns = ["name", "month", "grade"]
more_grades["bonus"] = [np.nan, np.nan, np.nan, 0, np.nan, 2, 3, 3, 0, 0, 1, 0]
more_grades

# --- notebook cell 243 ---
pd.pivot_table(more_grades, index="name")

# --- notebook cell 245 ---
pd.pivot_table(more_grades, index="name", values=["grade","bonus"], aggfunc=np.max)

# --- notebook cell 247 ---
pd.pivot_table(more_grades, index="name", values="grade", columns="month", margins=True)

# --- notebook cell 249 ---
pd.pivot_table(more_grades, index=("name", "month"), margins=True)

# --- notebook cell 251 ---
much_data = np.fromfunction(lambda x,y: (x+y*y)%17*11, (10000, 26))
large_df = pd.DataFrame(much_data, columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
large_df[large_df % 16 == 0] = np.nan
large_df.insert(3,"some_text", "Blabla")
large_df

# --- notebook cell 253 ---
large_df.head()

# --- notebook cell 255 ---
large_df.tail(n=2)

# --- notebook cell 257 ---
large_df.info()

# --- notebook cell 259 ---
large_df.describe()

# --- notebook cell 261 ---
my_df = pd.DataFrame(
    [["Biking", 68.5, 1985, np.nan], ["Dancing", 83.1, 1984, 3]], 
    columns=["hobby","weight","birthyear","children"],
    index=["alice", "bob"]
)
my_df

# --- notebook cell 263 ---
my_df.to_csv("my_df.csv")
my_df.to_html("my_df.html")
my_df.to_json("my_df.json")

# --- notebook cell 265 ---
for filename in ("my_df.csv", "my_df.html", "my_df.json"):
    print("#", filename)
    with open(filename, "rt") as f:
        print(f.read())
        print()

# --- notebook cell 267 ---
try:
    my_df.to_excel("my_df.xlsx", sheet_name='People')
except ImportError as e:
    print(e)

# --- notebook cell 269 ---
my_df_loaded = pd.read_csv("my_df.csv", index_col=0)
my_df_loaded

# --- notebook cell 271 ---
us_cities = None
try:
    csv_url = "http://simplemaps.com/files/cities.csv"
    us_cities = pd.read_csv(csv_url, index_col=0)
    us_cities = us_cities.head()
except IOError as e:
    print(e)
us_cities

# --- notebook cell 274 ---
city_loc = pd.DataFrame(
    [
        ["CA", "San Francisco", 37.781334, -122.416728],
        ["NY", "New York", 40.705649, -74.008344],
        ["FL", "Miami", 25.791100, -80.320733],
        ["OH", "Cleveland", 41.473508, -81.739791],
        ["UT", "Salt Lake City", 40.755851, -111.896657]
    ], columns=["state", "city", "lat", "lng"])
city_loc

# --- notebook cell 275 ---
city_pop = pd.DataFrame(
    [
        [808976, "San Francisco", "California"],
        [8363710, "New York", "New-York"],
        [413201, "Miami", "Florida"],
        [2242193, "Houston", "Texas"]
    ], index=[3,4,5,6], columns=["population", "city", "state"])
city_pop

# --- notebook cell 277 ---
pd.merge(left=city_loc, right=city_pop, on="city")

# --- notebook cell 279 ---
all_cities = pd.merge(left=city_loc, right=city_pop, on="city", how="outer")
all_cities

# --- notebook cell 281 ---
pd.merge(left=city_loc, right=city_pop, on="city", how="right")

# --- notebook cell 283 ---
city_pop2 = city_pop.copy()
city_pop2.columns = ["population", "name", "state"]
pd.merge(left=city_loc, right=city_pop2, left_on="city", right_on="name")

# --- notebook cell 285 ---
result_concat = pd.concat([city_loc, city_pop])
result_concat

# --- notebook cell 287 ---
result_concat.loc[3]

# --- notebook cell 289 ---
pd.concat([city_loc, city_pop], ignore_index=True)

# --- notebook cell 291 ---
pd.concat([city_loc, city_pop], join="inner")

# --- notebook cell 293 ---
pd.concat([city_loc, city_pop], axis=1)

# --- notebook cell 295 ---
pd.concat([city_loc.set_index("city"), city_pop.set_index("city")], axis=1)

# --- notebook cell 298 ---
city_loc.append(city_pop)

# --- notebook cell 301 ---
city_eco = city_pop.copy()
city_eco["eco_code"] = [17, 17, 34, 20]
city_eco

# --- notebook cell 303 ---
city_eco["economy"] = city_eco["eco_code"].astype('category')
city_eco["economy"].cat.categories

# --- notebook cell 305 ---
city_eco["economy"].cat.categories = ["Finance", "Energy", "Tourism"]
city_eco

# --- notebook cell 307 ---
city_eco.sort_values(by="economy", ascending=False)