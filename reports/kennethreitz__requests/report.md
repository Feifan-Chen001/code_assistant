# CodeAssistant 报告

## 1. 代码审查（Review）

### 概览

| 指标 | 值 |
| --- | --- |
| 问题总数 | 25 |
| 高/中/低 | 7/3/15 |
| 工具数 | 4 |
| DS 规则数 | 0 |

### 严重性分布

| 严重性 | 数量 |
| --- | --- |
| low | 15 |
| high | 7 |
| medium | 3 |

### 工具分布

| 工具 | 数量 |
| --- | --- |
| bandit | 14 |
| rule-plugin | 7 |
| pip-audit | 3 |
| ast-rule | 1 |

### Top 20 问题

| 严重性 | 工具 | 规则 | 位置 | 说明 |
| --- | --- | --- | --- | --- |
| high | ast-rule | BANNED_CALL | setup.py:79 | 禁止使用 exec（项目/领域规则），建议安全替代或白名单。 |
| low | rule-plugin | PY_LOOP_INVARIANT | src/requests/models.py:178 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | src/requests/models.py:180 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | src/requests/sessions.py:190 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_GLOBAL_VARIABLE | src/requests/status_codes.py:120 | 检测到全局变量 '__doc__' 的使用; 考虑使用参数传递代替 |
| low | rule-plugin | PY_LOOP_INVARIANT | src/requests/utils.py:586 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | src/requests/utils.py:643 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | rule-plugin | PY_LOOP_INVARIANT | src/requests/utils.py:645 | 考虑在循环外计算长度，避免每次迭代都重新计算 |
| low | bandit | B605 | D:\code_assistant\Git_repo\kennethreitz__requests\setup.py:57 | Starting a process with a shell: Seems safe, but may be changed in the future, consider rewriting without shell |
| medium | bandit | B607 | D:\code_assistant\Git_repo\kennethreitz__requests\setup.py:57 | Starting a process with a partial executable path |
| low | bandit | B605 | D:\code_assistant\Git_repo\kennethreitz__requests\setup.py:58 | Starting a process with a shell: Seems safe, but may be changed in the future, consider rewriting without shell |
| medium | bandit | B607 | D:\code_assistant\Git_repo\kennethreitz__requests\setup.py:58 | Starting a process with a partial executable path |
| medium | bandit | B102 | D:\code_assistant\Git_repo\kennethreitz__requests\setup.py:79 | Use of exec detected. |
| low | bandit | B101 | D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\__init__.py:60 | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. |
| low | bandit | B101 | D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\__init__.py:70 | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. |
| low | bandit | B101 | D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\__init__.py:72 | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. |
| low | bandit | B101 | D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\__init__.py:79 | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. |
| low | bandit | B101 | D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\__init__.py:84 | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. |
| low | bandit | B101 | D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\_internal_utils.py:45 | Use of assert detected. The enclosed code will be removed when compiling to optimised byte code. |
| high | bandit | B324 | D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\auth.py:148 | Use of weak MD5 hash for security. Consider usedforsecurity=False |

### 复杂度摘要（Radon）

```
D:\code_assistant\Git_repo\kennethreitz__requests\setup.py
    C 31:0 PyTest - A (2)
    M 34:4 PyTest.initialize_options - A (2)
    M 43:4 PyTest.finalize_options - A (1)
    M 48:4 PyTest.run_tests - A (1)
D:\code_assistant\Git_repo\kennethreitz__requests\docs\_themes\flask_theme_support.py
    C 7:0 FlaskyStyle - A (1)
D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\adapters.py
    M 502:4 HTTPAdapter.send - C (19)
    M 274:4 HTTPAdapter.cert_verify - C (14)
    F 75:0 _urllib3_request_context - B (6)
    M 435:4 HTTPAdapter.request_url - B (6)
    C 137:0 HTTPAdapter - A (5)
    M 367:4 HTTPAdapter._get_connection - A (4)
    M 236:4 HTTPAdapter.proxy_manager_for - A (3)
    M 396:4 HTTPAdapter.get_connection - A (3)
    C 107:0 BaseAdapter - A (2)
    M 172:4 HTTPAdapter.__init__ - A (2)
    M 194:4 HTTPAdapter.__getstate__ - A (2)
    M 197:4 HTTPAdapter.__setstate__ - A (2)
    M 330:4 HTTPAdapter.build_response - A (2)
    M 425:4 HTTPAdapter.close - A (2)
    M 481:4 HTTPAdapter.proxy_headers - A (2)
    F 61:4 SOCKSProxyManager - A (1)
    M 110:4 BaseAdapter.__init__ - A (1)
    M 113:4 BaseAdapter.send - A (1)
    M 132:4 BaseAdapter.close - A (1)
    M 210:4 HTTPAdapter.init_poolmanager - A (1)
    M 467:4 HTTPAdapter.add_headers - A (1)
D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\api.py
    F 14:0 request - A (1)
    F 62:0 get - A (1)
    F 76:0 options - A (1)
    F 88:0 head - A (1)
    F 103:0 post - A (1)
    F 118:0 put - A (1)
    F 133:0 patch - A (1)
    F 148:0 delete - A (1)
D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\auth.py
    M 126:4 HTTPDigestAuth.build_digest_header - C (19)
    F 25:0 _basic_auth_str - A (5)
    C 107:0 HTTPDigestAuth - A (5)
    M 241:4 HTTPDigestAuth.handle_401 - A (5)
    M 285:4 HTTPDigestAuth.__call__ - A (3)
    C 69:0 AuthBase - A (2)
    C 76:0 HTTPBasicAuth - A (2)
    C 99:0 HTTPProxyAuth - A (2)
    M 116:4 HTTPDigestAuth.init_per_thread_state - A (2)
    M 236:4 HTTPDigestAuth.handle_redirect - A (2)
    M 72:4 AuthBase.__call__ - A (1)
    M 79:4 HTTPBasicAuth.__init__ - A (1)
    M 83:4 HTTPBasicAuth.__eq__ - A (1)
    M 91:4 HTTPBasicAuth.__ne__ - A (1)
    M 94:4 HTTPBasicAuth.__call__ - A (1)
    M 102:4 HTTPProxyAuth.__call__ - A (1)
    M 110:4 HTTPDigestAuth.__init__ - A (1)
    M 305:4 HTTPDigestAuth.__eq__ - A (1)
    M 313:4 HTTPDigestAuth.__ne__ - A (1)
D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\cookies.py
    M 386:4 RequestsCookieJar._find_no_duplicates - B (9)
    F 151:0 remove_cookie_by_name - B (8)
    F 521:0 cookiejar_from_dict - B (7)
    M 366:4 RequestsCookieJar._find - B (7)
    F 542:0 merge_cookies - B (6)
    M 306:4 RequestsCookieJar.get_dict - B (6)
    F 492:0 morsel_to_cookie - A (5)
    F 440:0 _copy_cookie_jar - A (4)
    M 293:4 RequestsCookieJar.multiple_domains - A (4)
    M 349:4 RequestsCookieJar.set_cookie - A (4)
    F 124:0 extract_cookies_to_jar - A (3)
    C 176:0 RequestsCookieJar - A (3)
    M 206:4 RequestsCookieJar.set - A (3)
    M 277:4 RequestsCookieJar.list_domains - A (3)
    M 285:4 RequestsCookieJar.list_paths - A (3)
    M 358:4 RequestsCookieJar.update - A (3)
    F 455:0 create_cookie - A (2)
    C 23:0 MockRequest - A (2)
    M 49:4 MockRequest.get_full_url - A (2)
    M 72:4 MockRequest.has_header - A (2)
    C 103:0 MockResponse - A (2)
    M 194:4 RequestsCookieJar.get - A (2)
    M 225:4 RequestsCookieJar.iterkeys - A (2)
    M 242:4 RequestsCookieJar.itervalues - A (2)
    M 259:4 RequestsCookieJar.iteritems - A (2)
    M 321:4 RequestsCookieJar.__contains__ - A (2)
    M 422:4 RequestsCookieJar.__setstate__ - A (2)
    F 140:0 get_cookie_header - A (1)
    M 35:4 MockRequest.__init__ - A (1)
    M 40:4 MockRequest.get_type - A (1)
    M 43:4 MockRequest.get_host - A (1)
    M 46:4 MockRequest.get_origin_req_host - A (1)
    M 69:4 MockRequest.is_unverifiable - A (1)
    M 75:4 MockRequest.get_header - A (1)
    M 78:4 MockRequest.add_header - A (1)
    M 84:4 MockRequest.add_unredirected_header - A (1)
    M 87:4 MockRequest.get_new_headers - A (1)
    M 91:4 MockRequest.unverifiable - A (1)
    M 95:4 MockRequest.origin_req_host - A (1)
    M 99:4 MockRequest.host - A (1)
    M 110:4 MockResponse.__init__ - A (1)
    M 117:4 MockResponse.info - A (1)
    M 120:4 MockResponse.getheaders - A (1)
    C 170:0 CookieConflictError - A (1)
    M 234:4 RequestsCookieJar.keys - A (1)
    M 251:4 RequestsCookieJar.values - A (1)
    M 268:4 RequestsCookieJar.items - A (1)
    M 327:4 RequestsCookieJar.__getitem__ - A (1)
    M 336:4 RequestsCookieJar.__setitem__ - A (1)
    M 343:4 RequestsCookieJar.__delitem__ - A (1)
    M 415:4 RequestsCookieJar.__getstate__ - A (1)
    M 428:4 RequestsCookieJar.copy - A (1)
    M 435:4 RequestsCookieJar.get_policy - A (1)
D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\exceptions.py
    C 12:0 RequestException - A (5)
    M 17:4 RequestException.__init__ - A (4)
    C 31:0 JSONDecodeError - A (2)
    C 27:0 InvalidJSONError - A (1)
    M 34:4 JSONDecodeError.__init__ - A (1)
    M 44:4 JSONDecodeError.__reduce__ - A (1)
    C 55:0 HTTPError - A (1)
    C 59:0 ConnectionError - A (1)
    C 63:0 ProxyError - A (1)
    C 67:0 SSLError - A (1)
    C 71:0 Timeout - A (1)
    C 80:0 ConnectTimeout - A (1)
    C 87:0 ReadTimeout - A (1)
    C 91:0 URLRequired - A (1)
    C 95:0 TooManyRedirects - A (1)
    C 99:0 MissingSchema - A (1)
    C 103:0 InvalidSchema - A (1)
    C 107:0 InvalidURL - A (1)
    C 111:0 InvalidHeader - A (1)
    C 115:0 InvalidProxyURL - A (1)
    C 119:0 ChunkedEncodingError - A (1)
    C 123:0 ContentDecodingError - A (1)
    C 127:0 StreamConsumedError - A (1)
    C 131:0 RetryError - A (1)
    C 135:0 UnrewindableBodyError - A (1)
    C 142:0 RequestsWarning - A (1)
    C 146:0 FileModeWarning - A (1)
    C 150:0 RequestsDependencyWarning - A (1)
D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\help.py
    F 34:0 _implementation - B (6)
    F 69:0 info - B (6)
    F 128:0 main - A (1)
D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\hooks.py
    F 22:0 dispatch_hook - B (6)
    F 15:0 default_hooks - A (2)
D:\code_assistant\Git_repo\kennethreitz__requests\src\requests\models.py
    M 137:4 RequestEncodingMixin._encode_files - D (21)
    M 409:4 PreparedRequest.prepare_url - C (17)
    M 494:4 PreparedRequest.prepare_body - C (17)
    C 84:0 RequestEncodingMixin - C (13)
    M 107:4 RequestEncodingMixin._encode_params - C (11)
    M 852:4 Response.iter_lines - B (9)
    M 942:4 Response.json - B (8)
    M 258:4 Request.__init__ - B (7)
    M 794:4 Response.iter_content - B (7)
    M 207:4 RequestHooksMixin.register_hook - B (6)
    M 588:4 PreparedRequest.prepare_auth - B (6)
    M 886:4 Response.content - B (6)
    M 992:4 Response.raise_for_status - B (6)
    C 206:0 RequestHooksMixin - A (5)
    C 313:0 PreparedRequest - A (5)
    M 572:4 PreparedRequest.prepare_content_length - A (5)
    C 230:0 Request - A (4)
    C 640:0 Response - A (4)
    M 905:4 Response.text - A (4)
    M 976:4 Response.links - A (4)
    M 86:4 RequestEncodingMixin.path_url - A (3)
    M 483:4 PreparedRequest.prepare_headers - A (3)
    M 610:4 PreparedRequest.prepare_cookies - A (3)
    M 630:4 PreparedRequest.prepare_hooks - A (3)
    M 711:4 Response.__getstate__ - A (3)
    M 1021:4 Response.close - A (3)
    M 218:4 RequestHooksMixin.deregister_hook - A (2)
    M 382:4 PreparedRequest.copy - A (2)
    M 393:4 PreparedRequest.prepare_method - A (2)
    M 400:4 PreparedRequest._get_idna_encoded_host - A (2)
    M 719:4 Response.__setstate__ - A (2)
    M 755:4 Response.ok - A (2)
    M 770:4 Response.is_redirect - A (2)
    M 777:4 Response.is_permanent_redirect - A (2)
    M 292:4 Request.__repr__ - A (1)
    M 295:4 Request.prepare - A (1)
    M 334:4 PreparedRequest.__init__ - A (1)
    M 351:4 PreparedRequest.prepare - A (1)
    M 379:4 PreparedRequest.__repr__ - A (1)
    M 65
```

## 2. 测试生成（TestGen）

| 指标 | 值 |
| --- | --- |
| 写入测试文件数 | 8 |
| 覆盖函数数 | 63 |
| 输出目录 | D:\code_assistant\Git_repo\kennethreitz__requests\reports\kennethreitz__requests\generated_tests |

### 覆盖率报告（coverage report -m）

```
No data to report.

```
