
import pandas as pd

def inefficient_loading():
    # 问题：一次性加载大文件到内存
    df = pd.read_csv("huge_file.csv")
    return df.head()

def missing_error_handling():
    # 问题：没有错误处理
    df = pd.read_csv("data.csv")
    return df
