"""公共工具：把大模型返回的文本解析成 [0, hi] 的整数，带简单兜底。"""

import re

_EXPLICIT_RE = re.compile(r"(?:捕捞|捕获|捕|抓|捞)\s*[:：]?\s*(\d+)")
_NUM_RE = re.compile(r"\d+")


def parse_int(text, hi=50):
    """严格按“捕捞 N”解析；失败则取文本第一个数字；都没有则兜底为 0。

    返回 {"raw", "value", "valid", "reason"}，valid=False 表示走了兜底。
    """
    if not isinstance(text, str) or not text.strip():
        return {"raw": text, "value": 0, "valid": False, "reason": "empty"}

    m = _EXPLICIT_RE.search(text)
    if m and int(m.group(1)) <= hi:
        return {"raw": text, "value": int(m.group(1)), "valid": True, "reason": "explicit"}

    for tok in _NUM_RE.findall(text):
        if int(tok) <= hi:
            return {"raw": text, "value": int(tok), "valid": True, "reason": "regex"}

    return {"raw": text, "value": 0, "valid": False, "reason": "no_number"}
