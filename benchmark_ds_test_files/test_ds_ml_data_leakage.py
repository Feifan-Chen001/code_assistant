
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def data_leakage(X, y):
    # 问题：在split之前进行缩放，导致数据泄露
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
    return X_train, X_test, y_train, y_test
