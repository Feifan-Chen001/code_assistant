from __future__ import annotations
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def build_llm_config(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    data = cfg or {}
    llm_cfg = data.get("llm", {}) if isinstance(data, dict) else {}
    api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = llm_cfg.get("api_key") or os.getenv(api_key_env, "")
    return {
        "provider": str(llm_cfg.get("provider", "openai")),
        "model": str(llm_cfg.get("model", "gpt-4o-mini")),
        "api_key": api_key,
        "api_key_env": api_key_env,
        "base_url": str(llm_cfg.get("base_url", "https://api.openai.com/v1")),
        "temperature": float(llm_cfg.get("temperature", 0.2)),
        "max_tokens": int(llm_cfg.get("max_tokens", 1200)),
        "timeout": int(llm_cfg.get("timeout", 60)),
        "allow_empty_key": bool(llm_cfg.get("allow_empty_key", False)),
    }


def _extract_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    return str(msg.get("content") or "")


def llm_chat(messages: List[Dict[str, str]], cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    llm_cfg = build_llm_config(cfg)
    api_key = llm_cfg.get("api_key") or ""
    if not api_key and not llm_cfg.get("allow_empty_key"):
        return {"ok": False, "error": f"Missing API key. Set {llm_cfg.get('api_key_env')} or llm.api_key."}

    base_url = llm_cfg.get("base_url", "").rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    payload = {
        "model": llm_cfg.get("model"),
        "messages": messages,
        "temperature": llm_cfg.get("temperature", 0.2),
        "max_tokens": llm_cfg.get("max_tokens", 1200),
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=llm_cfg.get("timeout", 60)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw else {}
        text = _extract_text(data)
        if not text:
            return {"ok": False, "error": "Empty response from LLM.", "raw": data}
        return {"ok": True, "text": text, "raw": data}
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        return {"ok": False, "error": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
