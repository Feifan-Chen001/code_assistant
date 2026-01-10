# CodeAssistant 报告

## 1. 代码审查（Review）

### 概览

| 指标 | 值 |
| --- | --- |
| 问题总数 | 123 |
| 高/中/低 | 28/17/78 |
| 工具数 | 3 |
| DS 规则数 | 19 |

### 严重性分布

| 严重性 | 数量 |
| --- | --- |
| low | 78 |
| high | 28 |
| medium | 17 |

### 工具分布

| 工具 | 数量 |
| --- | --- |
| rule-plugin | 99 |
| ds-rule | 19 |
| ast-rule | 5 |

### DS 规则分布

| 规则 | 数量 |
| --- | --- |
| DS_PANDAS_SETTINGWITHCOPY | 19 |

### Top 20 问题

| 严重性 | 工具 | 规则 | 位置 | 说明 |
| --- | --- | --- | --- | --- |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | manage.py:141 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | rule-plugin | PY_RESOURCE_LEAK | manage.py:448 | open() 调用应在 with 语句中使用以确保正确关闭 |
| high | rule-plugin | PY_RESOURCE_LEAK | manage.py:190 | open() 调用应在 with 语句中使用以确保正确关闭 |
| high | rule-plugin | PY_RESOURCE_LEAK | manage.py:792 | open() 调用应在 with 语句中使用以确保正确关闭 |
| low | rule-plugin | PY_LOOP_INVARIANT | manage.py:532 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | manage.py:413 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | manage.py:418 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | manage.py:532 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | manage_api.py:141 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | rule-plugin | PY_RESOURCE_LEAK | manage_api.py:448 | open() 调用应在 with 语句中使用以确保正确关闭 |
| high | rule-plugin | PY_RESOURCE_LEAK | manage_api.py:190 | open() 调用应在 with 语句中使用以确保正确关闭 |
| high | rule-plugin | PY_RESOURCE_LEAK | manage_api.py:792 | open() 调用应在 with 语句中使用以确保正确关闭 |
| low | rule-plugin | PY_LOOP_INVARIANT | manage_api.py:532 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | manage_api.py:413 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | manage_api.py:418 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | manage_api.py:532 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | scripts/secmonkey_role_setup.py:211 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | security_monkey/auditor.py:422 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | security_monkey/auditor.py:423 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |
| high | ds-rule | DS_PANDAS_SETTINGWITHCOPY | security_monkey/auditor.py:424 | Chained indexing may trigger SettingWithCopy; use .loc/.iloc. |

### 复杂度摘要（Radon）

```
D:\code_assistant\Git_repo\Netflix__security_monkey\manage.py
    F 436:0 add_override_scores - D (27)
    F 355:0 add_override_score - C (15)
    F 710:0 sync_swag - C (13)
    F 572:0 _parse_accounts - B (7)
    F 781:0 sync_networks - B (7)
    F 185:0 amazon_accounts - B (6)
    F 230:0 create_user - A (5)
    F 112:0 delete_unjustified_issues - A (3)
    F 129:0 export_environment_summary - A (3)
    F 607:0 add_watcher_config - A (3)
    C 655:0 APIServer - A (3)
    M 672:4 APIServer.__call__ - A (3)
    C 820:0 AddAccount - A (3)
    M 839:4 AddAccount.__call__ - A (3)
    F 69:0 run_change_reporter - A (2)
    F 82:0 find_changes - A (2)
    F 98:0 audit_changes - A (2)
    F 151:0 backup_config_to_json - A (2)
    F 164:0 sync_jira - A (2)
    F 277:0 toggle_active_user - A (2)
    F 326:0 disable_accounts - A (2)
    F 338:0 enable_accounts - A (2)
    F 565:0 _parse_tech_names - A (2)
    F 862:0 main - A (2)
    M 826:4 AddAccount.get_options - A (2)
    F 63:0 drop_db - A (1)
    F 174:0 clear_expired_exceptions - A (1)
    F 596:0 delete_account - A (1)
    F 631:0 fetch_aws_canonical_ids - A (1)
    F 647:0 clean_stale_issues - A (1)
    M 656:4 APIServer.__init__ - A (1)
    M 660:4 APIServer.get_options - A (1)
    M 821:4 AddAccount.__init__ - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\manage_api.py
    F 436:0 add_override_scores - D (27)
    F 355:0 add_override_score - C (15)
    F 710:0 sync_swag - C (13)
    F 572:0 _parse_accounts - B (7)
    F 781:0 sync_networks - B (7)
    F 185:0 amazon_accounts - B (6)
    F 230:0 create_user - A (5)
    F 112:0 delete_unjustified_issues - A (3)
    F 129:0 export_environment_summary - A (3)
    F 607:0 add_watcher_config - A (3)
    C 655:0 APIServer - A (3)
    M 672:4 APIServer.handle - A (3)
    C 820:0 AddAccount - A (3)
    M 839:4 AddAccount.handle - A (3)
    F 69:0 run_change_reporter - A (2)
    F 82:0 find_changes - A (2)
    F 98:0 audit_changes - A (2)
    F 151:0 backup_config_to_json - A (2)
    F 164:0 sync_jira - A (2)
    F 277:0 toggle_active_user - A (2)
    F 326:0 disable_accounts - A (2)
    F 338:0 enable_accounts - A (2)
    F 565:0 _parse_tech_names - A (2)
    F 862:0 main - A (2)
    M 826:4 AddAccount.get_options - A (2)
    F 63:0 drop_db - A (1)
    F 174:0 clear_expired_exceptions - A (1)
    F 596:0 delete_account - A (1)
    F 631:0 fetch_aws_canonical_ids - A (1)
    F 647:0 clean_stale_issues - A (1)
    M 656:4 APIServer.__init__ - A (1)
    M 660:4 APIServer.get_options - A (1)
    M 821:4 AddAccount.__init__ - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\env-config\config-docker.py
    F 20:0 env_to_bool - A (3)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\env.py
    F 27:0 run_migrations_offline - A (1)
    F 45:0 run_migrations_online - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\00c1dabdbe85_lengthen_account_name.py
    F 17:0 upgrade - A (1)
    F 24:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\0ae4ef82b244_.py
    F 17:0 upgrade - A (1)
    F 23:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\11f081cf54e2_.py
    F 57:0 upgrade - B (6)
    F 102:0 downgrade - A (1)
    C 23:0 AccountType - A (1)
    C 32:0 Account - A (1)
    C 46:0 AccountTypeCustomValues - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\1583a48cb978_.py
    F 17:0 upgrade - A (1)
    F 25:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\15e39d43395f_.py
    F 17:0 upgrade - A (1)
    F 25:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\1727fb4309d8_.py
    F 17:0 upgrade - A (1)
    F 23:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\1a863bd1acb1_.py
    F 29:0 upgrade - A (2)
    F 38:0 downgrade - A (2)
    C 23:0 Technology - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\1c847ae1209a_.py
    F 17:0 upgrade - A (1)
    F 23:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\2705e6e13a8f_.py
    F 17:0 upgrade - A (1)
    F 30:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\2ce75615b24d_.py
    F 17:0 upgrade - A (1)
    F 30:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\2ea41f4610fd_.py
    F 17:0 upgrade - A (1)
    F 23:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\331ca47ce8ad_.py
    F 17:0 upgrade - A (1)
    F 21:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\4ac52090a637_.py
    F 17:0 upgrade - A (1)
    F 23:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\51170afa2b48_custom_role_name.py
    F 17:0 upgrade - A (1)
    F 23:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\538eeb160af6_.py
    F 35:0 upgrade - A (3)
    F 49:0 downgrade - A (1)
    C 28:0 User - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\55725cc4bf25_.py
    F 46:0 update_custom_value - A (3)
    F 63:0 update_from_custom_value - A (2)
    F 76:0 upgrade - A (2)
    F 95:0 downgrade - A (2)
    C 26:0 Account - A (1)
    C 35:0 AccountTypeCustomValues - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\57f648d4b597_.py
    F 17:0 upgrade - A (1)
    F 33:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\595e27f36454_.py
    F 16:0 upgrade - A (1)
    F 26:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\5bd631a1b748_.py
    F 30:0 upgrade - A (1)
    F 42:0 downgrade - A (1)
    C 24:0 ItemAudit - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\61a6fd4b4500_.py
    F 17:0 upgrade - A (1)
    F 27:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\6245d75fa12_exceptions_table.py
    F 17:0 upgrade - A (1)
    F 45:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\67ea2aac5ea0_.py
    F 17:0 upgrade - A (1)
    F 32:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\6b9d673d8e30_added_index_for_itemrevision_created.py
    F 17:0 upgrade - A (1)
    F 21:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\6d2354fb841c_.py
    F 17:0 upgrade - A (1)
    F 32:0 downgrade - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\7c54b06e227b_.py
    F 207:0 upgrade - B (6)
    F 260:0 downgrade - A (1)
    C 27:0 AccountType - A (1)
    C 36:0 Account - A (1)
    C 50:0 AccountTypeCustomValues - A (1)
    C 61:0 ExceptionLogs - A (1)
    C 89:0 ItemAudit - A (1)
    C 113:0 Item - A (1)
    C 136:0 ItemComment - A (1)
    C 148:0 ItemRevisionComment - A (1)
    C 160:0 ItemRevision - A (1)
    C 175:0 CloudTrailEntry - A (1)
    C 198:0 Technology - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\908b0085d28d_.py
    F 45:0 upgrade - A (4)
    C 25:0 User - A (2)
    F 67:0 downgrade - A (1)
    M 41:4 User.__str__ - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\a9fe9c93ed75_.py
    F 44:0 remove_duplicate_issue_items - A (3)
    F 66:0 remove_duplicate_association - A (3)
    F 88:0 remove_duplicate_role_users - A (3)
    F 110:0 upgrade - A (1)
    F 142:0 downgrade - A (1)
    C 23:0 IssueItemAssociation - A (1)
    C 30:0 AssociationTable - A (1)
    C 37:0 RolesUsers - A (1)
D:\code_assistant\Git_repo\Netflix__security_monkey\migrations\versions\ad23a56abf25_.py
    F 17:0 upgrade - A (1)
    F 28:0 downgrade - A (1)
D:\
```

## 2. 测试生成（TestGen）

| 指标 | 值 |
| --- | --- |
| 写入测试文件数 | 55 |
| 覆盖函数数 | 200 |
| 输出目录 | D:\code_assistant\Git_repo\Netflix__security_monkey\reports\Netflix__security_monkey\generated_tests |

### 覆盖率报告（coverage report -m）

```
No data to report.

```
