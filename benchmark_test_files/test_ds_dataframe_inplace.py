
import pandas as pd

def process_data(df):
    # 问题：使用inplace=True
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.fillna(0, inplace=True)
    return df

def chain_operations(df):
    # 问题：链式调用可能导致SettingWithCopyWarning
    df[df['age'] > 18]['score'] = 100
    return df
