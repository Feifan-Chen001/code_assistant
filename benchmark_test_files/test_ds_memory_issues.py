
import pandas as pd
import numpy as np

def memory_inefficient():
    # 问题：创建不必要的副本
    df = pd.DataFrame({'a': range(1000000)})
    df2 = df.copy()
    df3 = df.copy()
    df4 = df.copy()
    return df4

def concat_in_loop():
    # 问题：在循环中拼接DataFrame
    result = pd.DataFrame()
    for i in range(100):
        temp = pd.DataFrame({'col': [i]})
        result = pd.concat([result, temp])
    return result
