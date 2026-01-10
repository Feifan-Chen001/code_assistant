# CodeAssistant 报告

## 1. 代码审查（Review）

### 概览

| 指标 | 值 |
| --- | --- |
| 问题总数 | 28 |
| 高/中/低 | 28/0/0 |
| 工具数 | 1 |
| DS 规则数 | 0 |

### 严重性分布

| 严重性 | 数量 |
| --- | --- |
| high | 28 |

### 工具分布

| 工具 | 数量 |
| --- | --- |
| pip-audit | 28 |

### Top 20 问题

| 严重性 | 工具 | 规则 | 位置 | 说明 |
| --- | --- | --- | --- | --- |
| high | pip-audit | VULN |  | None None -> PYSEC-2021-421 Babel.Locale in Babel before 2.9.1 allows attackers to load arbitrary locale .dat files (containing serialized Python objects) via directory traversal,… |
| high | pip-audit | VULN |  | None None -> PYSEC-2022-42986 Certifi is a curated collection of Root Certificates for validating the trustworthiness of SSL certificates while verifying the identity of TLS hosts… |
| high | pip-audit | VULN |  | None None -> PYSEC-2023-135 Certifi 2023.07.22 removes root certificates from "e-Tugra" from the root store. These are in the process of being removed from Mozilla's trust store. … |
| high | pip-audit | VULN |  | None None -> PYSEC-2024-60 A vulnerability was identified in the kjd/idna library, specifically within the `idna.encode()` function, affecting version 3.6. The issue arises from t… |
| high | pip-audit | VULN |  | None None -> PYSEC-2021-66 This affects the package jinja2 from 0.0.0 and before 2.11.3. The ReDoS vulnerability is mainly due to the `_punctuation_re regex` operator and its use … |
| high | pip-audit | VULN |  | None None -> PYSEC-2019-217 In Pallets Jinja before 2.10.1, str.format_map allows a sandbox escape. |
| high | pip-audit | VULN |  | None None -> CVE-2024-22195 The `xmlattr` filter in affected versions of Jinja accepts keys containing spaces. XML/HTML attributes cannot contain spaces, as each would then be int… |
| high | pip-audit | VULN |  | None None -> CVE-2024-34064 The `xmlattr` filter in affected versions of Jinja accepts keys containing non-attribute characters. XML/HTML attributes cannot contain spaces, `/`, `>… |
| high | pip-audit | VULN |  | None None -> CVE-2024-56326 An oversight in how the Jinja sandboxed environment detects calls to `str.format` allows an attacker that controls the content of a template to execute… |
| high | pip-audit | VULN |  | None None -> CVE-2025-27516 An oversight in how the Jinja sandboxed environment interacts with the `|attr` filter allows an attacker that controls the content of a template to exe… |
| high | pip-audit | VULN |  | None None -> PYSEC-2021-140 An infinite loop in SMLLexer in Pygments versions 1.5 to 2.7.3 may lead to denial of service when performing syntax highlighting of a Standard ML (SML)… |
| high | pip-audit | VULN |  | None None -> PYSEC-2021-141 In pygments 1.1+, fixed in 2.7.4, the lexers used to parse programming languages rely heavily on regular expressions. Some of the regular expressions h… |
| high | pip-audit | VULN |  | None None -> PYSEC-2023-117 A ReDoS issue was discovered in pygments/lexers/smithy.py in pygments through 2.15.0 via SmithyLexer. |
| high | pip-audit | VULN |  | None None -> PYSEC-2018-28 The Requests package before 2.20.0 for Python sends an HTTP Authorization header to an http URI upon receiving a same-hostname https-to-http redirect, w… |
| high | pip-audit | VULN |  | None None -> PYSEC-2023-74 Requests is a HTTP library. Since Requests 2.3.0, Requests has been leaking Proxy-Authorization headers to destination servers when redirected to an HTT… |
| high | pip-audit | VULN |  | None None -> CVE-2024-35195 When making requests through a Requests `Session`, if the first request is made with `verify=False` to disable cert verification, all subsequent reques… |
| high | pip-audit | VULN |  | None None -> CVE-2024-47081 ### Impact  Due to a URL parsing issue, Requests releases prior to 2.32.4 may leak .netrc credentials to third parties for specific maliciously-crafted… |
| high | pip-audit | VULN |  | None None -> PYSEC-2021-108 An issue was discovered in urllib3 before 1.26.5. When provided with a URL containing many @ characters in the authority component, the authority regul… |
| high | pip-audit | VULN |  | None None -> PYSEC-2019-133 The urllib3 library before 1.24.2 for Python mishandles certain cases where the desired set of CA certificates is different from the OS store of CA cer… |
| high | pip-audit | VULN |  | None None -> PYSEC-2019-132 In the urllib3 library through 1.24.1 for Python, CRLF injection is possible if the attacker controls the request parameter. |

### 复杂度摘要（Radon）

```
D:\code_assistant\Git_repo\realpython__python-guide\test_issues.py
    F 37:0 process_list - A (3)
    F 4:0 append_to_list - A (1)
    F 9:0 add_to_dict - A (1)
    F 17:0 increment_counter - A (1)
    F 24:0 read_file_bad - A (1)
    F 31:0 read_file_good - A (1)
D:\code_assistant\Git_repo\realpython__python-guide\docs\_themes\flask_theme_support.py
    C 7:0 FlaskyStyle - A (1)

7 blocks (classes, functions, methods) analyzed.
Average complexity: A (1.2857142857142858)

```

## 2. 测试生成（TestGen）

| 指标 | 值 |
| --- | --- |
| 写入测试文件数 | 0 |
| 覆盖函数数 | 0 |
| 输出目录 | D:\code_assistant\Git_repo\realpython__python-guide\reports\realpython__python-guide\generated_tests |

### 覆盖率报告（coverage report -m）

```
No data to report.

```
