"""公共工具：OpenAI 兼容的 LLM 客户端（基于 requests），带瞬时错误重试。

可通过环境变量配置（API Key 必须提供，不内置默认值）：
    LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
"""

import os
import time

import requests

DEFAULT_BASE_URL = os.environ.get("LLM_BASE_URL", "http://llmapi.evolventapi.com/v1")
DEFAULT_API_KEY = os.environ.get("LLM_API_KEY", "")
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4.1-flash")


class LLMClient:
    def __init__(self, base_url=None, api_key=None, model=None,
                 timeout=120, retries=3, backoff=5, delay=0):
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or DEFAULT_API_KEY
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.delay = delay

    def chat(self, messages, temperature=0.7):
        if not self.api_key:
            raise RuntimeError("未设置 LLM_API_KEY 环境变量，无法调用 LLM")
        if self.delay:
            time.sleep(self.delay)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages, "temperature": temperature}

        for attempt in range(1, self.retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except requests.RequestException:
                if attempt == self.retries:
                    raise
                time.sleep(self.backoff * attempt)
