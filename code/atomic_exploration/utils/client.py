import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class Reply: # 回复统一格式
    content: str
    usage: dict | None
    model: str
    finish_reason: str | None = None


def safe_error_code(exc):
    """Return only adapter-owned codes; never persist provider text or secrets."""
    code = str(exc)
    if re.fullmatch(r"http_[1-5][0-9]{2}", code) or code in {
            "transport_error_billing_unknown", "invalid_response_billing_unknown"}:
        return code
    return "unknown_error"


class ApiError(RuntimeError):
    def __init__(self, message=""):
        super().__init__(message)
        self.error_code = safe_error_code(self)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Do not forward Authorization to another destination.


class ChatClient:
    def __init__(self, config, *, model=None):
        self.config = config
        self.key = os.environ.get("SOCIALEVO_API_KEY", "")
        self.model = model if model is not None else os.environ.get("SOCIALEVO_MODEL", "")
        base = os.environ.get("SOCIALEVO_BASE_URL", "").rstrip("/")
        parsed = urllib.parse.urlsplit(base)
        if not self.key or not self.model or not base:
            raise ValueError("Set SOCIALEVO_API_KEY, SOCIALEVO_MODEL, SOCIALEVO_BASE_URL locally")
        local = parsed.hostname in ("127.0.0.1", "localhost", "::1")
        if (parsed.scheme != "https" and not (local and parsed.scheme == "http")) or not parsed.hostname:
            raise ValueError("API base must use HTTPS (HTTP allowed only for local tests)")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Do not put credentials, query parameters or fragments in the base URL")
        self.url = base + "/chat/completions"
        self.opener = urllib.request.build_opener(NoRedirect())

    def complete(self, messages):
        payload = {"model": self.model, "messages": messages}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        payload[self.config.token_parameter] = self.config.max_output_tokens
        reasoning_effort = getattr(self.config, "reasoning_effort", None)
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        req = urllib.request.Request(self.url, json.dumps(payload).encode("utf-8"),
                                     {"Authorization": f"Bearer {self.key}",
                                      "Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=self.config.timeout) as response:
                data = json.loads(response.read(2_000_001))
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if content is None and choice.get("finish_reason") == "length":
                content = ""
            if not isinstance(content, str):
                raise ValueError("Missing text content")
            usage = data.get("usage")
            if not isinstance(usage, dict):
                usage = None
            return Reply(content, usage, data.get("model", self.model), choice.get("finish_reason"))
        except urllib.error.HTTPError as exc:
            raise ApiError(f"http_{exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ApiError("transport_error_billing_unknown") from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise ApiError("invalid_response_billing_unknown") from None


class MockClient:
    """
    冒烟测试，无 API ，确定性的测试 bot，用来验收环境，通信和评分链路是否正确
    """
    model = "scripted-smoke-test-NOT-LLM"
    key = ""

    def complete(self, messages):
        obs = json.loads(messages[-1]["content"])
        possible = set(obs["private_possible_answers"])
        for message in obs["inbox"]:
            try:
                possible &= set(json.loads(message["content"])["possible"])
            except (ValueError, KeyError, TypeError):
                pass
        if obs["communication_allowed"] and obs["round"] == 1:
            action = {"type": "send", "content": json.dumps({"possible": sorted(possible)})}
        else:
            action = {"type": "submit", "answer": sorted(possible)[0]}
        return Reply(json.dumps(action), None, self.model, "stop")
