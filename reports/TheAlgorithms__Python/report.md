# CodeAssistant 报告

## 1. 代码审查（Review）

### 概览

| 指标 | 值 |
| --- | --- |
| 问题总数 | 961 |
| 高/中/低 | 305/6/650 |
| 工具数 | 2 |
| DS 规则数 | 308 |

### 严重性分布

| 严重性 | 数量 |
| --- | --- |
| low | 650 |
| high | 305 |
| medium | 6 |

### 工具分布

| 工具 | 数量 |
| --- | --- |
| rule-plugin | 653 |
| ds-rule | 308 |

### DS 规则分布

| 规则 | 数量 |
| --- | --- |
| DS_PANDAS_SETTINGWITHCOPY | 249 |
| DS_RANDOM_SEED | 50 |
| DS_HYPERPARAMS_HARDCODED | 4 |
| DS_SKLEARN_RANDOM_STATE | 3 |
| DS_PIPELINE_SUGGEST | 2 |

### Top 20 问题

| 严重性 | 工具 | 规则 | 位置 | 说明 |
| --- | --- | --- | --- | --- |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/all_permutations.py:68 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/combination_sum.py:34 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/combination_sum.py:38 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | backtracking/crossword_puzzle_solver.py:54 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | backtracking/crossword_puzzle_solver.py:56 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | backtracking/crossword_puzzle_solver.py:77 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | backtracking/crossword_puzzle_solver.py:79 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/crossword_puzzle_solver.py:27 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/crossword_puzzle_solver.py:29 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/crossword_puzzle_solver.py:31 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/crossword_puzzle_solver.py:75 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/crossword_puzzle_solver.py:106 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/crossword_puzzle_solver.py:107 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/crossword_puzzle_solver.py:107 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/generate_parentheses_iterative.py:46 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | backtracking/hamiltonian_cycle.py:98 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | backtracking/knight_tour.py:63 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | backtracking/knight_tour.py:66 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | backtracking/knight_tour.py:89 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | backtracking/knight_tour.py:92 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |

### 复杂度摘要（Radon）

```
D:\code_assistant\Git_repo\TheAlgorithms__Python\audio_filters\butterworth_filter.py
    F 13:0 make_lowpass - A (1)
    F 43:0 make_highpass - A (1)
    F 73:0 make_bandpass - A (1)
    F 104:0 make_allpass - A (1)
    F 131:0 make_peak - A (1)
    F 163:0 make_lowshelf - A (1)
    F 200:0 make_highshelf - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\audio_filters\iir_filter.py
    M 39:4 IIRFilter.set_coefficients - A (4)
    C 4:0 IIRFilter - A (3)
    M 75:4 IIRFilter.process - A (2)
    M 26:4 IIRFilter.__init__ - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\audio_filters\show_response.py
    F 38:0 show_frequency_response - A (2)
    F 70:0 show_phase_response - A (2)
    C 11:0 FilterType - A (2)
    F 22:0 get_bounds - A (1)
    M 13:4 FilterType.process - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\all_combinations.py
    F 23:0 generate_all_combinations - A (3)
    F 64:0 create_all_state - A (3)
    F 13:0 combination_lists - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\all_permutations.py
    F 16:0 create_state_space_tree - A (4)
    F 12:0 generate_all_permutations - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\all_subsequences.py
    F 18:0 create_state_space_tree - A (2)
    F 14:0 generate_all_subsequences - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\coloring.py
    F 36:0 util_color - A (5)
    F 10:0 valid_coloring - A (3)
    F 88:0 color - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\combination_sum.py
    F 16:0 backtrack - A (4)
    F 41:0 combination_sum - A (4)
    F 68:0 main - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\crossword_puzzle_solver.py
    F 82:0 solve_crossword - B (8)
    F 4:0 is_valid - B (7)
    F 36:0 place_word - A (3)
    F 59:0 remove_word - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\generate_parentheses.py
    F 11:0 backtrack - A (4)
    F 48:0 generate_parenthesis - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\generate_parentheses_iterative.py
    F 1:0 generate_parentheses_iterative - A (5)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\hamiltonian_cycle.py
    F 49:0 util_hamilton_cycle - A (5)
    F 11:0 valid_connection - A (3)
    F 110:0 hamilton_cycle - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\knight_tour.py
    F 71:0 open_knight_tour - B (6)
    F 49:0 open_knight_tour_helper - A (5)
    F 6:0 get_valid_pos - A (4)
    F 35:0 is_complete - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\match_word_pattern.py
    F 1:0 match_word_pattern - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\minimax.py
    F 16:0 minimax - A (5)
    F 81:0 main - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\n_queens.py
    F 16:0 is_safe - B (6)
    F 55:0 solve - A (4)
    F 84:0 printboard - A (4)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\n_queens_math.py
    F 82:0 depth_first_search - B (7)
    F 141:0 n_queens_solution - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\power_sum.py
    F 10:0 backtrack - A (4)
    F 54:0 solve - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\rat_in_maze.py
    F 139:0 run_maze - C (14)
    F 4:0 solve_maze - B (8)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\sudoku.py
    F 44:0 is_safe - B (6)
    F 75:0 sudoku - A (5)
    F 63:0 find_empty_location - A (4)
    F 112:0 print_solution - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\sum_of_subsets.py
    F 32:0 create_state_space_tree - A (5)
    F 11:0 generate_sum_of_subsets_solutions - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\word_break.py
    F 10:0 backtrack - A (5)
    F 48:0 word_break - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\word_ladder.py
    F 15:0 backtrack - B (6)
    F 68:0 word_ladder - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\backtracking\word_search.py
    F 91:0 word_exists - C (14)
    F 47:0 exits_word - B (8)
    F 36:0 get_point_key - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\binary_and_operator.py
    F 4:0 binary_and - A (5)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\binary_coded_decimal.py
    F 1:0 binary_coded_decimal - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\binary_count_setbits.py
    F 1:0 binary_count_setbits - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\binary_count_trailing_zeros.py
    F 4:0 binary_count_trailing_zeros - A (4)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\binary_or_operator.py
    F 4:0 binary_or - A (4)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\binary_shifts.py
    F 36:0 logical_right_shift - A (4)
    F 6:0 logical_left_shift - A (3)
    F 68:0 arithmetic_right_shift - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\binary_twos_complement.py
    F 4:0 twos_complement - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\binary_xor_operator.py
    F 4:0 binary_xor - A (4)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\bitwise_addition_recursive.py
    F 7:0 bitwise_addition_recursive - B (6)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\count_1s_brian_kernighan_method.py
    F 1:0 get_1s_count - A (4)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\count_number_of_one_bits.py
    F 33:0 get_set_bits_count_using_modulo_operator - A (4)
    F 4:0 get_set_bits_count_using_brian_kernighans_algorithm - A (3)
    F 63:0 benchmark - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\excess_3_code.py
    F 1:0 excess_3_code - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\find_previous_power_of_two.py
    F 1:0 find_previous_power_of_two - B (6)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\find_unique_number.py
    F 1:0 find_unique_number - A (5)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\gray_code_sequence.py
    F 50:0 gray_code_sequence_string - A (5)
    F 1:0 gray_code - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\highest_set_bit.py
    F 1:0 get_highest_set_bit_position - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\index_of_rightmost_set_bit.py
    F 4:0 get_index_of_rightmost_set_bit - A (4)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\is_even.py
    F 1:0 is_even - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\is_power_of_two.py
    F 18:0 is_power_of_two - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\largest_pow_of_two_le_num.py
    F 22:0 largest_pow_of_two_le_num - A (4)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\missing_number.py
    F 1:0 find_missing_number - A (2)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\numbers_different_signs.py
    F 14:0 different_signs - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\power_of_4.py
    F 18:0 power_of_4 - A (5)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\reverse_bits.py
    F 31:0 reverse_bit - A (4)
    F 1:0 get_reverse_bit_string - A (3)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\single_bit_manipulation_operations.py
    F 6:0 set_bit - A (1)
    F 24:0 clear_bit - A (1)
    F 40:0 flip_bit - A (1)
    F 56:0 is_bit_set - A (1)
    F 77:0 get_bit - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\bit_manipulation\swap_all_odd_and_even_bits.py
    F 1:0 show_bits - A (1)
    F 10:0 swap_odd_even_bits - A (1)
D:\code_assistant\Git_repo\TheAlgorithms__Python\blockc
```

## 2. 测试生成（TestGen）

| 指标 | 值 |
| --- | --- |
| 写入测试文件数 | 93 |
| 覆盖函数数 | 200 |
| 输出目录 | D:\code_assistant\Git_repo\TheAlgorithms__Python\reports\TheAlgorithms__Python\generated_tests |

### 覆盖率报告（coverage report -m）

```
No data to report.

```
