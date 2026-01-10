
import pandas as pd

def inefficient_apply(df):
    # 问题：对简单操作使用apply
    df['new_col'] = df['col1'].apply(lambda x: x * 2)
    
    # 更好的方式
    # df['new_col'] = df['col1'] * 2
    return df
