"""DS规则单元测试"""
import pytest
from pathlib import Path
from src.features.review.ds_rules import scan_source_ds, _DSVisitor


class TestRandomSeedDetection:
    """测试随机性与种子检测规则"""

    def test_random_usage_without_seed(self):
        """应检测到未设置种子的随机操作"""
        code = """
import random
x = random.random()
"""
        findings = scan_source_ds(code, "test.py")
        assert any(f.rule == "DS_RANDOM_SEED" for f in findings)

    def test_random_with_seed(self):
        """设置种子后不应报警"""
        code = """
import random
random.seed(42)
x = random.random()
"""
        findings = scan_source_ds(code, "test.py")
        assert not any(f.rule == "DS_RANDOM_SEED" for f in findings)

    def test_numpy_random_without_seed(self):
        """检测 numpy.random 未设置种子"""
        code = """
import numpy as np
x = np.random.randn(10)
"""
        findings = scan_source_ds(code, "test.py")
        assert any(f.rule == "DS_RANDOM_SEED" for f in findings)

    def test_numpy_random_with_seed(self):
        """numpy.random.seed 应被识别"""
        code = """
import numpy as np
np.random.seed(42)
x = np.random.randn(10)
"""
        findings = scan_source_ds(code, "test.py")
        assert not any(f.rule == "DS_RANDOM_SEED" for f in findings)


class TestDataLeakageDetection:
    """测试数据泄漏检测规则"""

    def test_fit_transform_before_split(self):
        """检测 fit_transform 在 train_test_split 之前"""
        code = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X = load_iris().data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
"""
        findings = scan_source_ds(code, "test.py")
        assert any(f.rule == "DS_LEAKAGE_FIT_BEFORE_SPLIT" for f in findings)

    def test_fit_transform_after_split_ok(self):
        """split 后再 fit_transform 不应报警"""
        code = """
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = [[1, 2], [3, 4]]
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""
        findings = scan_source_ds(code, "test.py")
        assert not any(f.rule == "DS_LEAKAGE_FIT_BEFORE_SPLIT" for f in findings)


class TestPipelineUsage:
    """测试 Pipeline 建议规则"""

    def test_scaler_without_pipeline(self):
        """检测缩放器未在 Pipeline 中使用"""
        code = """
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
"""
        findings = scan_source_ds(code, "test.py")
        assert any(f.rule == "DS_PIPELINE_SUGGEST" for f in findings)

    def test_scaler_in_pipeline(self):
        """Pipeline 中使用缩放器不应报警"""
        code = """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
])
"""
        findings = scan_source_ds(code, "test.py")
        assert not any(f.rule == "DS_PIPELINE_SUGGEST" for f in findings)


class TestSklearnRandomState:
    """测试 sklearn 随机状态检测"""

    def test_random_forest_without_random_state(self):
        """检测 RandomForest 缺少 random_state"""
        code = """
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
"""
        findings = scan_source_ds(code, "test.py")
        assert any(f.rule == "DS_SKLEARN_RANDOM_STATE" for f in findings)

    def test_random_forest_with_random_state(self):
        """指定 random_state 不应报警"""
        code = """
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
"""
        findings = scan_source_ds(code, "test.py")
        assert not any(f.rule == "DS_SKLEARN_RANDOM_STATE" for f in findings)


class TestPandasPerformance:
    """测试 pandas 性能检测"""

    def test_iterrows_detected(self):
        """检测低效的 iterrows"""
        code = """
for idx, row in df.iterrows():
    print(row['value'])
"""
        findings = scan_source_ds(code, "test.py")
        assert any(f.rule == "DS_PANDAS_ITERROWS" for f in findings)

    def test_apply_axis1_detected(self):
        """检测低效的 apply(axis=1)"""
        code = """
result = df.apply(lambda x: x['a'] + x['b'], axis=1)
"""
        findings = scan_source_ds(code, "test.py")
        assert any(f.rule == "DS_PANDAS_APPLY_AXIS1" for f in findings)


class TestModelSerialization:
    """测试模型序列化检测"""

    def test_pickle_dump_model(self):
        """检测使用 pickle 序列化模型"""
        code = """
import pickle
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
"""
        findings = scan_source_ds(code, "test.py")
        assert any(f.rule == "DS_MODEL_PICKLE_UNSAFE" for f in findings)


class TestHardcodedHyperparameters:
    """测试硬编码超参数检测"""

    def test_hardcoded_params(self):
        """检测硬编码的模型参数"""
        code = """
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5)
"""
        findings = scan_source_ds(code, "test.py")
        assert any(f.rule == "DS_HYPERPARAMS_HARDCODED" for f in findings)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
