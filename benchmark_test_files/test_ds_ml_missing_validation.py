
from sklearn.linear_model import LogisticRegression

def train_without_validation(X_train, y_train, X_test, y_test):
    # 问题：直接在测试集上评估，没有验证集
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score
