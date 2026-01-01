# CodeAssistant 报告

## 1. 代码审查（Review）

### 概览

| 指标 | 值 |
| --- | --- |
| 问题总数 | 251 |
| 高/中/低 | 25/124/102 |
| 工具数 | 2 |
| DS 规则数 | 226 |

### 严重性分布

| 严重性 | 数量 |
| --- | --- |
| medium | 124 |
| low | 102 |
| high | 25 |

### 工具分布

| 工具 | 数量 |
| --- | --- |
| ds-rule | 226 |
| ast-rule | 25 |

### DS 规则分布

| 规则 | 数量 |
| --- | --- |
| DS_PANDAS_APPLY_AXIS1 | 86 |
| DS_PANDAS_SETTINGWITHCOPY | 82 |
| DS_RANDOM_SEED | 42 |
| DS_PANDAS_ITERROWS | 16 |

### Top 20 问题

| 严重性 | 工具 | 规则 | 位置 | 说明 |
| --- | --- | --- | --- | --- |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/algorithms.py:40 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/arithmetic.py:51 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/array.py:122 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/attrs_caching.py:14 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/boolean.py:9 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/categoricals.py:75 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/ctors.py:79 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/dtypes.py:85 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/eval.py:16 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/frame_ctor.py:33 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/frame_methods.py:43 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/gil.py:95 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/groupby.py:96 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/hash_functions.py:83 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/indexing.py:42 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/inference.py:27 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/join_merge.py:31 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/libs.py:65 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/multiindex_object.py:78 | Randomness detected without an explicit seed. |
| medium | ds-rule | DS_RANDOM_SEED | pandas/asv_bench/benchmarks/plotting.py:42 | Randomness detected without an explicit seed. |

### 复杂度摘要（Radon）

```
D:\�·ɷ�\��ѧ\�����ھ�\����ҵ\CodeAssistant\my_repo\pandas\generate_pxi.py
    F 16:0 main - A (2)
    F 7:0 process_tempita - A (1)
D:\�·ɷ�\��ѧ\�����ھ�\����ҵ\CodeAssistant\my_repo\pandas\generate_version.py
    F 32:0 main - A (5)
    F 13:0 write_version_info - A (3)
D:\�·ɷ�\��ѧ\�����ھ�\����ҵ\CodeAssistant\my_repo\pandas\setup.py
    M 112:4 CleanCommand.initialize_options - B (10)
    F 394:0 maybe_cythonize - B (6)
    C 107:0 CleanCommand - B (6)
    C 76:0 build_ext - A (5)
    M 78:4 build_ext.render_templates - A (5)
    M 167:4 CleanCommand.run - A (5)
    M 233:4 CheckSDist.run - A (5)
    C 185:0 CheckSDist - A (4)
    C 252:0 CheckingBuildExt - A (4)
    M 257:4 CheckingBuildExt.check_cython_extensions - A (4)
    M 98:4 build_ext.build_extensions - A (2)
    C 274:0 CythonCommand - A (2)
    C 285:0 DummyBuildSrc - A (2)
    F 31:0 is_platform_windows - A (1)
    F 35:0 is_platform_mac - A (1)
    F 427:0 srcpath - A (1)
    M 164:4 CleanCommand.finalize_options - A (1)
    M 230:4 CheckSDist.initialize_options - A (1)
    M 269:4 CheckingBuildExt.build_extensions - A (1)
    M 281:4 CythonCommand.build_extension - A (1)
    M 290:4 DummyBuildSrc.initialize_options - A (1)
    M 293:4 DummyBuildSrc.finalize_options - A (1)
    M 296:4 DummyBuildSrc.run - A (1)
D:\�·ɷ�\��ѧ\�����ھ�\����ҵ\CodeAssistant\my_repo\pandas\asv_bench\benchmarks\algorithms.py
    M 34:4 Factorize.setup - C (11)
    M 85:4 Duplicated.setup - B (9)
    C 68:0 Duplicated - B (6)
    C 15:0 Factorize - A (5)
    C 111:0 DuplicatedMaskedArray - A (3)
    C 176:0 Quantile - A (3)
    M 184:4 Quantile.setup - A (3)
    M 119:4 DuplicatedMaskedArray.setup - A (2)
    C 133:0 Hashing - A (2)
    M 134:4 Hashing.setup_cache - A (2)
    C 198:0 SortIntegerArray - A (2)
    M 61:4 Factorize.time_factorize - A (1)
    M 64:4 Factorize.peakmem_factorize - A (1)
    M 107:4 Duplicated.time_duplicated - A (1)
    M 129:4 DuplicatedMaskedArray.time_duplicated - A (1)
    M 154:4 Hashing.time_frame - A (1)
    M 157:4 Hashing.time_series_int - A (1)
    M 160:4 Hashing.time_series_string - A (1)
    M 163:4 Hashing.time_series_float - A (1)
    M 166:4 Hashing.time_series_categorical - A (1)
    M 169:4 Hashing.time_series_timedeltas - A (1)
    M 172:4 Hashing.time_series_dates - A (1)
    M 194:4 Quantile.time_quantile - A (1)
    M 201:4 SortIntegerArray.setup - A (1)
    M 206:4 SortIntegerArray.time_argsort - A (1)
D:\�·ɷ�\��ѧ\�����ھ�\����ҵ\CodeAssistant\my_repo\pandas\asv_bench\benchmarks\arithmetic.py
    M 178:4 Ops.setup - A (3)
    M 323:4 IndexArithmetic.setup - A (3)
    C 454:0 BinaryOpsMultiIndex - A (3)
    C 28:0 IntFrameWithScalar - A (2)
    C 58:0 OpWithFillValue - A (2)
    C 75:0 MixedFrameWithSeriesAxis - A (2)
    C 113:0 FrameWithFrameWide - A (2)
    M 133:4 FrameWithFrameWide.setup - A (2)
    C 174:0 Ops - A (2)
    C 204:0 Ops2 - A (2)
    C 257:0 Timeseries - A (2)
    C 286:0 IrregularOps - A (2)
    C 298:0 TimedeltaOps - A (2)
    C 307:0 CategoricalComparisons - A (2)
    C 319:0 IndexArithmetic - A (2)
    C 346:0 NumericInferOps - A (2)
    C 373:0 DateInferOps - A (2)
    C 422:0 OffsetArrayArithmetic - A (2)
    C 441:0 ApplyIndex - A (2)
    M 458:4 BinaryOpsMultiIndex.setup - A (2)
    M 50:4 IntFrameWithScalar.setup - A (1)
    M 54:4 IntFrameWithScalar.time_frame_op_with_scalar - A (1)
    M 59:4 OpWithFillValue.setup - A (1)
    M 68:4 OpWithFillValue.time_frame_op_with_fill_value_no_nas - A (1)
    M 71:4 OpWithFillValue.time_series_op_with_fill_value_no_nas - A (1)
    M 94:4 MixedFrameWithSeriesAxis.setup - A (1)
    M 102:4 MixedFrameWithSeriesAxis.time_frame_op_with_series_axis0 - A (1)
    M 105:4 MixedFrameWithSeriesAxis.time_frame_op_with_series_axis1 - A (1)
    M 165:4 FrameWithFrameWide.time_op_different_blocks - A (1)
    M 169:4 FrameWithFrameWide.time_op_same_blocks - A (1)
    M 187:4 Ops.time_frame_add - A (1)
    M 190:4 Ops.time_frame_mult - A (1)
    M 193:4 Ops.time_frame_multi_and - A (1)
    M 196:4 Ops.time_frame_comparison - A (1)
    M 199:4 Ops.teardown - A (1)
    M 205:4 Ops2.setup - A (1)
    M 225:4 Ops2.time_frame_float_div - A (1)
    M 228:4 Ops2.time_frame_float_div_by_zero - A (1)
    M 231:4 Ops2.time_frame_float_floor_by_zero - A (1)
    M 234:4 Ops2.time_frame_int_div_by_zero - A (1)
    M 239:4 Ops2.time_frame_int_mod - A (1)
    M 242:4 Ops2.time_frame_float_mod - A (1)
    M 247:4 Ops2.time_frame_dot - A (1)
    M 250:4 Ops2.time_series_dot - A (1)
    M 253:4 Ops2.time_frame_series_dot - A (1)
    M 261:4 Timeseries.setup - A (1)
    M 270:4 Timeseries.time_series_timestamp_compare - A (1)
    M 273:4 Timeseries.time_series_timestamp_different_reso_compare - A (1)
    M 276:4 Timeseries.time_timestamp_series_compare - A (1)
    M 279:4 Timeseries.time_timestamp_ops_diff - A (1)
    M 282:4 Timeseries.time_timestamp_ops_diff_with_shift - A (1)
    M 287:4 IrregularOps.setup - A (1)
    M 294:4 IrregularOps.time_add - A (1)
    M 299:4 TimedeltaOps.setup - A (1)
    M 303:4 TimedeltaOps.time_add_td_ts - A (1)
    M 311:4 CategoricalComparisons.setup - A (1)
    M 315:4 CategoricalComparisons.time_categorical_op - A (1)
    M 330:4 IndexArithmetic.time_add - A (1)
    M 333:4 IndexArithmetic.time_subtract - A (1)
    M 336:4 IndexArithmetic.time_multiply - A (1)
    M 339:4 IndexArithmetic.time_divide - A (1)
    M 342:4 IndexArithmetic.time_modulo - A (1)
    M 351:4 NumericInferOps.setup - A (1)
    M 357:4 NumericInferOps.time_add - A (1)
    M 360:4 NumericInferOps.time_subtract - A (1)
    M 363:4 NumericInferOps.time_multiply - A (1)
    M 366:4 NumericInferOps.time_divide - A (1)
    M 369:4 NumericInferOps.time_modulo - A (1)
    M 375:4 DateInferOps.setup_cache - A (1)
    M 381:4 DateInferOps.time_subtract_datetimes - A (1)
    M 384:4 DateInferOps.time_timedelta_plus_datetime - A (1)
    M 387:4 DateInferOps.time_add_timedeltas - A (1)
    M 426:4 OffsetArrayArithmetic.setup - A (1)
    M 432:4 OffsetArrayArithmetic.time_add_series_offset - A (1)
    M 436:4 OffsetArrayArithmetic.time_add_dti_offset - A (1)
    M 445:4 ApplyIndex.setup - A (1)
    M 450:4 ApplyIndex.time_apply_index - A (1)
    M 475:4 BinaryOpsMultiIndex.time_binary_op_multiindex - A (1)
D:\�·ɷ�\��ѧ\�����ھ�\����ҵ\CodeAssistant\my_repo\pandas\asv_bench\benchmarks\array.py
    M 119:4 ArrowExtensionArray.setup - B (8)
    C 106:0 ArrowExtensionArray - B (6)
    M 76:4 ArrowStringArray.setup - A (5)
    C 45:0 IntervalArray - A (3)
    C 72:0 ArrowStringArray - A (3)
    C 6:0 BooleanArray - A (2)
    C 31:0 IntegerArray - A (2)
    M 46:4 IntervalArray.setup - A (2)
    C 54:0 StringArray - A (2)
    M 55:4 StringArray.setup - A (2)
    M 88:4 ArrowStringArray.time_setitem - A (2)
    M 7:4 BooleanArray.setup - A (1)
    M 15:4 BooleanArray.time_constructor - A (1)
    M 18:4 BooleanArray.time_from_bool_array - A (1)
    M 21:4 BooleanArray.time_from_integer_array - A (1)
    M 24:4 BooleanArray.time_from_integer_like - A (1)
    M 27:4 BooleanArray.time_from_float_array - A (1)
    M 32:4 IntegerArray.setup - A (1)
    M 38:4 IntegerArray.time_constructor - A (1)
    M 41:4 IntegerArray.time_from_integer_array - A (1)
    M 50:4 IntervalArray.time_from_tuples - A (1)
    M 62:4 StringArray.time_from_np_object_array - A (1)
    M 65:4 StringArray.time_from_np_str_array - A (1)
    M 68:4 StringArray.time_from_list - A (1)
    M 92:4 ArrowStringArray.time_setitem_list - A (1)
    M 96:4 ArrowStringArray.time_setitem_slice - A (1)
    M 99:4 ArrowStringArray.time_setitem_null_slice - A (1)
    M 102:4 ArrowStringArray.time_tolist - A (1)
    M 139:4 ArrowExtensionArray.time_to_numpy - A (1)
D:\�·ɷ�\��ѧ\�����ھ�\����ҵ\CodeAssistant\my_repo\pandas\asv_bench\benchmarks\attrs_caching.py
    M 28:4 SeriesArrayAttribute.setup - B (6)
    C 24:0 SeriesArrayAttribute - A (3)
    C 12:0 DataFrameAttributes - A (2)
    M 13:4 DataFrameAttributes.setup - A (1)
    M 17:4 DataFrameAttributes.time_get_index - A (1)
    M 20:4 DataFrameAttribute
```

## 2. 测试生成（TestGen）

| 指标 | 值 |
| --- | --- |
| 写入测试文件数 | 20 |
| 覆盖函数数 | 200 |
| 输出目录 | D:\陈飞帆\大学\数据挖掘\大作业\CodeAssistant\generated_tests |

### 覆盖率报告（coverage report -m）

```
Name                                                         Stmts   Miss  Cover   Missing
------------------------------------------------------------------------------------------
pandas\pandas\__init__.py                                       36     25    31%   46-238
pandas\pandas\_config\__init__.py                               14      6    57%   34-35, 39-40, 44-45
pandas\pandas\_config\config.py                                276    162    41%   128-140, 187-191, 269-288, 332-341, 385-398, 402-403, 416-425, 428-439, 442, 500-518, 558, 560, 571, 573, 580, 586, 638-643, 657-665, 669-673, 684-689, 700, 708-712, 723-746, 751-773, 850, 871, 875, 881-894, 911-919, 944-946
pandas\pandas\_config\dates.py                                   7      0   100%
pandas\pandas\_config\display.py                                24      7    71%   27-28, 32-38, 42
pandas\pandas\_libs\__init__.py                                  5      3    40%   17-19
pandas\pandas\_typing.py                                       152      0   100%
pandas\pandas\compat\__init__.py                                30     12    60%   49-52, 64, 76, 88, 100, 112, 126, 138, 151
pandas\pandas\compat\_constants.py                              15      0   100%
pandas\pandas\compat\_optional.py                               44     33    25%   77-84, 148-191
pandas\pandas\compat\numpy\__init__.py                          24      6    75%   18, 37-42
pandas\pandas\compat\pyarrow.py                                 27     11    59%   22-32
pandas\pandas\core\__init__.py                                   0      0   100%
pandas\pandas\core\config_init.py                              203    196     3%   35-915
pandas\pandas\errors\__init__.py                                78     73     6%   18-1034
pandas\pandas\util\__init__.py                                  19     17    11%   3-25, 29
pandas\pandas\util\_exceptions.py                               50     38    24%   23-34, 43-63, 87-101
pandas\pandas\util\version\__init__.py                         201     80    60%   24, 27, 30, 33, 36, 39, 42, 45, 53, 56, 59, 62, 65, 68, 71, 74, 108, 139, 146, 151-154, 157-160, 164, 169-172, 175-178, 225, 250, 253-278, 290, 294, 302-305, 309, 317, 326, 330, 334, 338, 342, 346, 355-373, 377-379, 389, 418, 424, 431, 438, 451
pandas\scripts\__init__.py                                       0      0   100%
pandas\scripts\check_for_inconsistent_pandas_namespace.py       62     44    29%   45-46, 49-54, 57-59, 63-83, 89-117, 121-135, 139
pandas\scripts\check_test_naming.py                             65     50    23%   31-35, 39-41, 50, 56, 70, 80-138, 142-156
pandas\scripts\generate_pip_deps_from_conda.py                  63     51    19%   48-66, 92-126, 130-154
pandas\scripts\sort_whatsnew_note.py                            34     25    26%   44-62, 66-78, 82
pandas\scripts\tests\__init__.py                                 0      0   100%
pandas\scripts\tests\conftest.py                                 2      0   100%
pandas\scripts\tests\test_check_test_naming.py                   8      4    50%   37-40
pandas\scripts\tests\test_inconsistent_namespace_check.py       24      8    67%   38-41, 48, 53-57
pandas\scripts\tests\test_sort_whatsnew_note.py                  6      4    33%   5-30
pandas\scripts\tests\test_validate_docstrings.py                92     88     4%   10-530
pandas\scripts\tests\test_validate_exception_location.py        23     11    52%   32, 38-45, 51-54, 58-59
pandas\scripts\tests\test_validate_min_versions_in_sync.py      20     12    40%   47-58
pandas\scripts\tests\test_validate_unwanted_patterns.py         39     21    46%   56-60, 225-229, 257-259, 296-298, 315-332
pandas\scripts\validate_docstrings.py                          218    206     6%   31-536
pandas\scripts\validate_exception_location.py                   47     32    32%   43-47, 52-54, 57-70, 76-88, 92-101, 105
pandas\scripts\validate_min_versions_in_sync.py                196    163    17%   55-75, 79-84, 88-100, 106-126, 132-146, 152-179, 184-190, 196-227, 232-249, 253-281, 285
pandas\scripts\validate_unwanted_patterns.py                   124    105    15%   88-95, 115-147, 164-177, 207-286, 310-340, 363-391, 430-442, 446-473
pandas\web\pandas_web.py                                       225    210     7%   43-506
pandas\web\tests\test_pandas_web.py                             30     26    13%   12-88
tests\test_math_utils.py                                         9      7    22%   4-12
------------------------------------------------------------------------------------------
TOTAL                                                         2492   1736    30%

```
