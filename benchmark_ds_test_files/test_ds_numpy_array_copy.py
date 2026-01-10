
import numpy as np

def missing_copy(arr):
    # 问题：没有复制，可能意外修改原数组
    new_arr = arr
    new_arr[0] = 999
    return new_arr

def should_copy(arr):
    # 问题：缺少.copy()
    filtered = arr[arr > 0]
    filtered *= 2
    return filtered
