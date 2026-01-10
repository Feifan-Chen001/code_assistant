# CodeAssistant 报告

## 1. 代码审查（Review）

### 概览

| 指标 | 值 |
| --- | --- |
| 问题总数 | 195 |
| 高/中/低 | 13/117/65 |
| 工具数 | 4 |
| DS 规则数 | 114 |

### 严重性分布

| 严重性 | 数量 |
| --- | --- |
| medium | 117 |
| low | 65 |
| high | 13 |

### 工具分布

| 工具 | 数量 |
| --- | --- |
| ds-rule | 114 |
| rule-plugin | 74 |
| bandit | 6 |
| ast-rule | 1 |

### DS 规则分布

| 规则 | 数量 |
| --- | --- |
| DS_HYPERPARAMS_HARDCODED | 92 |
| DS_PIPELINE_SUGGEST | 10 |
| DS_RANDOM_SEED | 8 |
| DS_SKLEARN_RANDOM_STATE | 3 |
| DS_PANDAS_SETTINGWITHCOPY | 1 |

### Top 20 问题

| 严重性 | 工具 | 规则 | 位置 | 说明 |
| --- | --- | --- | --- | --- |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 01_the_machine_learning_landscape.ipynb#cell-52:3 | Model 'model' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| high | ds-rule | DS_RANDOM_SEED | 02_end_to_end_machine_learning_project.ipynb#cell-17:5 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_PIPELINE_SUGGEST | 02_end_to_end_machine_learning_project.ipynb#cell-81:2 | ordinal_encoder used without Pipeline; consider sklearn.pipeline.Pipeline. |
| medium | ds-rule | DS_PIPELINE_SUGGEST | 02_end_to_end_machine_learning_project.ipynb#cell-84:8 | cat_encoder used without Pipeline; consider sklearn.pipeline.Pipeline. |
| medium | ds-rule | DS_PIPELINE_SUGGEST | 02_end_to_end_machine_learning_project.ipynb#cell-88:2 | cat_encoder used without Pipeline; consider sklearn.pipeline.Pipeline. |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 02_end_to_end_machine_learning_project.ipynb#cell-120:3 | Model 'tree_reg' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 02_end_to_end_machine_learning_project.ipynb#cell-127:3 | Model 'forest_reg' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 02_end_to_end_machine_learning_project.ipynb#cell-131:3 | Model 'svm_reg' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 02_end_to_end_machine_learning_project.ipynb#cell-132:10 | Model 'forest_reg' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 02_end_to_end_machine_learning_project.ipynb#cell-139:9 | Model 'forest_reg' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| high | ds-rule | DS_RANDOM_SEED | 03_classification.ipynb#cell-20:3 | Randomness detected without an explicit seed. |
| low | rule-plugin | PY_LOOP_INVARIANT | 03_classification.ipynb#cell-27:16 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 03_classification.ipynb#cell-60:2 | Model 'forest_clf' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| medium | ds-rule | DS_PIPELINE_SUGGEST | 03_classification.ipynb#cell-77:3 | scaler used without Pipeline; consider sklearn.pipeline.Pipeline. |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 03_classification.ipynb#cell-98:2 | Model 'knn_clf' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 03_classification.ipynb#cell-169:3 | Model 'svm_clf' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 03_classification.ipynb#cell-176:3 | Model 'forest_clf' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |
| medium | ast-rule | BARE_EXCEPT | 03_classification.ipynb#cell-216:9 | Avoid bare except; catch explicit exception types. |
| low | rule-plugin | PY_LOOP_INVARIANT | 03_classification.ipynb#cell-223:22 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| medium | ds-rule | DS_HYPERPARAMS_HARDCODED | 03_classification.ipynb#cell-235:4 | Model 'log_clf' has hardcoded hyperparameters; consider using GridSearchCV or extracting to config. |

### 复杂度摘要（Radon）

```
D:\code_assistant\Git_repo\ageron__handson-ml\future_encoders.py
    M 553:4 OneHotEncoder._legacy_fit_transform - C (15)
    F 1483:0 _get_column_indices - C (14)
    M 455:4 OneHotEncoder._handle_deprecations - C (14)
    M 162:4 _BaseEncoder._fit - C (13)
    C 155:0 _BaseEncoder - C (11)
    F 1427:0 _get_column - B (10)
    M 726:4 OneHotEncoder.inverse_transform - B (10)
    M 1301:4 ColumnTransformer.fit_transform - B (10)
    M 635:4 OneHotEncoder._legacy_transform - B (9)
    M 1114:4 ColumnTransformer._iter - B (9)
    F 103:0 _transform_selected - B (8)
    F 1396:0 _check_key_type - B (8)
    M 1140:4 ColumnTransformer._validate_transformers - B (7)
    M 1160:4 ColumnTransformer._validate_remainder - B (7)
    M 200:4 _BaseEncoder._transform - B (6)
    M 1196:4 ColumnTransformer.get_feature_names - B (6)
    M 1221:4 ColumnTransformer._update_fitted_transformers - B (6)
    F 85:0 _handle_zeros_in_scale - A (5)
    C 242:0 OneHotEncoder - A (5)
    M 676:4 OneHotEncoder._transform_new - A (5)
    C 938:0 ColumnTransformer - A (5)
    M 1258:4 ColumnTransformer._fit_transform - A (5)
    F 1535:0 _get_transformer_list - A (4)
    M 380:4 OneHotEncoder.__init__ - A (4)
    M 896:4 OrdinalEncoder.inverse_transform - A (4)
    M 1246:4 ColumnTransformer._validate_output - A (4)
    M 1379:4 ColumnTransformer._hstack - A (4)
    F 55:0 _fit_transform_one - A (3)
    M 526:4 OneHotEncoder.fit - A (3)
    M 610:4 OneHotEncoder.fit_transform - A (3)
    C 794:0 OrdinalEncoder - A (3)
    F 47:0 _transform_one - A (2)
    F 1548:0 make_column_transformer - A (2)
    M 706:4 OneHotEncoder.transform - A (2)
    M 1071:4 ColumnTransformer._transformers - A (2)
    M 1081:4 ColumnTransformer._transformers - A (2)
    M 1184:4 ColumnTransformer.named_transformers_ - A (2)
    M 1351:4 ColumnTransformer.transform - A (2)
    F 43:0 _fit_one_transformer - A (1)
    F 81:0 _argmax - A (1)
    M 409:4 OneHotEncoder.n_values - A (1)
    M 415:4 OneHotEncoder.n_values - A (1)
    M 421:4 OneHotEncoder.categorical_features - A (1)
    M 427:4 OneHotEncoder.categorical_features - A (1)
    M 435:4 OneHotEncoder.active_features_ - A (1)
    M 442:4 OneHotEncoder.feature_indices_ - A (1)
    M 449:4 OneHotEncoder.n_values_ - A (1)
    M 858:4 OrdinalEncoder.__init__ - A (1)
    M 862:4 OrdinalEncoder.fit - A (1)
    M 879:4 OrdinalEncoder.transform - A (1)
    M 1062:4 ColumnTransformer.__init__ - A (1)
    M 1086:4 ColumnTransformer.get_params - A (1)
    M 1102:4 ColumnTransformer.set_params - A (1)
    M 1278:4 ColumnTransformer.fit - A (1)
D:\code_assistant\Git_repo\ageron__handson-ml\docker\jupyter_notebook_config.py
    F 4:0 export_script_and_view - A (4)
D:\code_assistant\Git_repo\ageron__handson-ml\docker\bin\nbclean_checkpoints
    M 52:4 NotebookAnalyser.clean_checkpoints - B (10)
    C 10:0 NotebookAnalyser - A (5)
    M 12:4 NotebookAnalyser.__init__ - A (3)
    M 86:4 NotebookAnalyser.clean_checkpoints_recursively - A (3)
    F 94:0 main - A (2)
    M 31:4 NotebookAnalyser.get_hash - A (2)
    M 47:4 NotebookAnalyser.log - A (1)
D:\code_assistant\Git_repo\ageron__handson-ml\docker\bin\rm_empty_subdirs
    F 5:0 remove_empty_directories - C (15)
    F 40:0 main - A (2)

64 blocks (classes, functions, methods) analyzed.
Average complexity: A (4.640625)

```

## 2. 测试生成（TestGen）

| 指标 | 值 |
| --- | --- |
| 写入测试文件数 | 16 |
| 覆盖函数数 | 138 |
| 输出目录 | D:\code_assistant\Git_repo\ageron__handson-ml\reports\ageron__handson-ml\generated_tests |

### 覆盖率报告（coverage report -m）

```
No data to report.

```
