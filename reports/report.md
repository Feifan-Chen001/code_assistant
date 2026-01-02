# CodeAssistant 报告

## 1. 代码审查（Review）

### 概览

| 指标 | 值 |
| --- | --- |
| 问题总数 | 62 |
| 高/中/低 | 60/0/2 |
| 工具数 | 2 |
| DS 规则数 | 0 |

### 严重性分布

| 严重性 | 数量 |
| --- | --- |
| high | 60 |
| low | 2 |

### 工具分布

| 工具 | 数量 |
| --- | --- |
| pip-audit | 60 |
| bandit | 2 |

### Top 20 问题

| 严重性 | 工具 | 规则 | 位置 | 说明 |
| --- | --- | --- | --- | --- |
| low | bandit | B101 | D:\code\PythonWithPycharm\DataMining\code_assistant\my_repo\tests\test_math_utils.py:5 | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. |
| low | bandit | B101 | D:\code\PythonWithPycharm\DataMining\code_assistant\my_repo\tests\test_math_utils.py:8 | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. |
| high | pip-audit | VULN |  | None None -> CVE-2024-27306 ### Summary  A XSS vulnerability exists on index pages for static file handling.  ### Details  When using `web.static(..., show_index=True)`, the resul… |
| high | pip-audit | VULN |  | None None -> CVE-2024-30251 ### Summary An attacker can send a specially crafted POST (multipart/form-data) request. When the aiohttp server processes it, the server will enter an… |
| high | pip-audit | VULN |  | None None -> CVE-2024-52304 ### Summary The Python parser parses newlines in chunk extensions incorrectly which can lead to request smuggling vulnerabilities under certain conditi… |
| high | pip-audit | VULN |  | None None -> CVE-2025-53643 ### Summary The Python parser is vulnerable to a request smuggling vulnerability due to not parsing trailer sections of an HTTP request.  ### Impact If… |
| high | pip-audit | VULN |  | None None -> CVE-2025-6176 Scrapy versions up to 2.13.3 are vulnerable to a denial of service (DoS) attack due to a flaw in its brotli decompression implementation. The protection… |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-225 cryptography is a package designed to expose cryptographic primitives and recipes to Python developers. Starting in version 38.0.0 and prior to version… |
| high | pip-audit | VULN |  | None None -> GHSA-h4gh-qq45-vh27 pyca/cryptography's wheels include a statically linked copy of OpenSSL. The versions of OpenSSL included in cryptography 37.0.0-43.0.0 are vulnera… |
| high | pip-audit | VULN |  | None None -> CVE-2024-12797 pyca/cryptography's wheels include a statically linked copy of OpenSSL. The versions of OpenSSL included in cryptography 42.0.0-44.0.0 are vulnerable t… |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-58 An issue was discovered in Django 5.0 before 5.0.7 and 4.2 before 4.2.14. Derived classes of the django.core.files.storage.Storage base class, when they… |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-57 An issue was discovered in Django 5.0 before 5.0.7 and 4.2 before 4.2.14. The django.contrib.auth.backends.ModelBackend.authenticate() method allows rem… |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-56 An issue was discovered in Django 4.2 before 4.2.14 and 5.0 before 5.0.7. urlize and urlizetrunc were subject to a potential denial of service attack vi… |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-59 An issue was discovered in Django 5.0 before 5.0.7 and 4.2 before 4.2.14. get_supported_language_variant() was subject to a potential denial-of-service … |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-69 An issue was discovered in Django 5.0 before 5.0.8 and 4.2 before 4.2.15. The urlize and urlizetrunc template filters, and the AdminURLFieldWidget widge… |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-70 An issue was discovered in Django 5.0 before 5.0.8 and 4.2 before 4.2.15. QuerySet.values() and values_list() methods on models with a JSONField are sub… |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-68 An issue was discovered in Django 5.0 before 5.0.8 and 4.2 before 4.2.15. The urlize() and urlizetrunc() template filters are subject to a potential den… |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-67 An issue was discovered in Django 5.0 before 5.0.8 and 4.2 before 4.2.15. The floatformat template filter is subject to significant memory consumption w… |
| high | pip-audit | VULN |  | None None -> PYSEC-2025-13 An issue was discovered in Django 5.1 before 5.1.7, 5.0 before 5.0.13, and 4.2 before 4.2.20. The django.utils.text.wrap() method and wordwrap template … |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-102 An issue was discovered in Django 5.1 before 5.1.1, 5.0 before 5.0.9, and 4.2 before 4.2.16. The urlize() and urlizetrunc() template filters are subjec… |

### 复杂度摘要（Radon）

```
D:\code\PythonWithPycharm\DataMining\code_assistant\my_repo\tests\test_math_utils.py
    F 4:0 test_add - A (2)
    F 7:0 test_div - A (2)
    F 10:0 test_div_zero - A (1)

3 blocks (classes, functions, methods) analyzed.
Average complexity: A (1.6666666666666667)

```

## 2. 测试生成（TestGen）

| 指标 | 值 |
| --- | --- |
| 写入测试文件数 | 4 |
| 覆盖函数数 | 7 |
| 输出目录 | D:\code\PythonWithPycharm\DataMining\code_assistant\generated_tests |

### 覆盖率报告（coverage report -m）

```
Name                       Stmts   Miss  Cover   Missing
--------------------------------------------------------
tests\test_math_utils.py       9      7    22%   4-12
--------------------------------------------------------
TOTAL                          9      7    22%

```
