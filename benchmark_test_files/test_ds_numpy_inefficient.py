
import numpy as np

def slow_loop(arr):
    # 问题：使用Python循环而不是向量化
    result = []
    for x in arr:
        result.append(x ** 2)
    return np.array(result)

def nested_loop(arr1, arr2):
    # 问题：嵌套循环可以向量化
    result = np.zeros((len(arr1), len(arr2)))
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            result[i, j] = arr1[i] + arr2[j]
    return result
