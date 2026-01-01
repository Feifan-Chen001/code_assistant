"""高级DS规则单元测试"""
import pytest
from src.features.review.ds_rules_advanced import scan_source_advanced_ds


class TestFeatureSelectionNestedCV:
    """测试特征选择嵌套CV检测"""

    def test_feature_selection_without_cv(self):
        """检测特征选择未使用嵌套CV"""
        code = """
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import load_iris

X = load_iris().data
selector = SelectKBest(k=5)
X_selected = selector.fit_transform(X, y)
"""
        findings = scan_source_advanced_ds(code, "test.py")
        assert any(f.rule == "DS_FEATURE_SELECTION_NO_NESTED_CV" for f in findings)

    def test_feature_selection_with_cv(self):
        """特征选择加CV不应报警"""
        code = """
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import cross_val_score

selector = SelectKBest(k=5)
scores = cross_val_score(selector, X, y)
"""
        findings = scan_source_advanced_ds(code, "test.py")
        assert not any(f.rule == "DS_FEATURE_SELECTION_NO_NESTED_CV" for f in findings)


class TestImbalanceHandling:
    """测试不平衡数据处理"""

    def test_imbalance_outside_pipeline(self):
        """检测采样方法未在Pipeline中"""
        code = """
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
smote = SMOTE()
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
"""
        findings = scan_source_advanced_ds(code, "test.py")
        assert any(f.rule == "DS_IMBALANCE_NOT_IN_PIPELINE" for f in findings)

    def test_imbalance_in_pipeline(self):
        """Pipeline中使用采样不应报警"""
        code = """
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('smote', SMOTE()),
    ('clf', LogisticRegression())
])
"""
        findings = scan_source_advanced_ds(code, "test.py")
        assert not any(f.rule == "DS_IMBALANCE_NOT_IN_PIPELINE" for f in findings)


class TestEvaluationMetrics:
    """测试评估指标检测"""

    def test_insufficient_metrics(self):
        """检测评估指标不足"""
        code = """
from sklearn.metrics import accuracy_score
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
"""
        findings = scan_source_advanced_ds(code, "test.py")
        assert any(f.rule == "DS_EVALUATION_INCOMPLETE" for f in findings)

    def test_comprehensive_metrics(self):
        """使用多个指标不应报警"""
        code = """
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
"""
        findings = scan_source_advanced_ds(code, "test.py")
        assert not any(f.rule == "DS_EVALUATION_INCOMPLETE" for f in findings)


class TestNoImbalanceHandling:
    """测试不平衡处理缺失检测"""

    def test_unhandled_imbalance(self):
        """检测训练无平衡措施"""
        code = """
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
"""
        findings = scan_source_advanced_ds(code, "test.py")
        assert any(f.rule == "DS_IMBALANCE_UNHANDLED" for f in findings)

    def test_class_weight_handling(self):
        """使用 class_weight 不应报警"""
        code = """
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
"""
        findings = scan_source_advanced_ds(code, "test.py")
        assert not any(f.rule == "DS_IMBALANCE_UNHANDLED" for f in findings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
