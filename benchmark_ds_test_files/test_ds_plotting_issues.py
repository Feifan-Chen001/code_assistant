
import matplotlib.pyplot as plt

def missing_close():
    # 问题：没有关闭图形，可能导致内存泄漏
    plt.figure()
    plt.plot([1, 2, 3], [1, 2, 3])
    plt.savefig("plot.png")
    # Missing plt.close()

def poor_defaults():
    # 问题：没有设置合适的图形参数
    plt.plot([1, 2, 3])
    # 缺少标题、标签、图例
