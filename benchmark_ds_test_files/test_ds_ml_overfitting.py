
from sklearn.ensemble import RandomForestClassifier

def overfitting_model(X, y):
    # 问题：没有正则化，容易过拟合
    model = RandomForestClassifier(
        n_estimators=1000,
        max_depth=None,  # 无限深度
        min_samples_split=2,  # 最小分裂样本太小
        min_samples_leaf=1
    )
    model.fit(X, y)
    return model
