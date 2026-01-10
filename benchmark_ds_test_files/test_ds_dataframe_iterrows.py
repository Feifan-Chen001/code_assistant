
import pandas as pd

def slow_iteration(df):
    # 问题：使用iterrows效率低
    total = 0
    for idx, row in df.iterrows():
        total += row['value']
    return total

def better_way(df):
    # 更好的方式
    return df['value'].sum()
