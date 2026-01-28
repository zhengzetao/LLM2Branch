import json
import re


# class LLMReplyParserBase(object):
#     def __init__(self) -> None:
#         pass

#     def parse_llm_reply(self, llm_reply):
#         pass


# class LLMReplyParser(LLMReplyParserBase):
#     def __init__(self) -> None:
#         super().__init__()

#     def parse_llm_reply(self, llm_reply: str):
#         success, json_obj = self.extract_json_from_text_string(llm_reply)
#         if not success:
#             return False, None
#         return success, json_obj

#     def extract_json_from_text_string(self, text_str: str):
#         json_strs = re.findall(r'\{.*?\}', text_str)
#         for json_str in json_strs:
#             try:
#                 json_obj = json.loads(json_str)
#                 result = self.get_cands_selected(json_obj)
#                 if result is not None:
#                     return True, result
#             except Exception as e:
#                 continue
#         return False, None

#     def get_cands_selected(self, json_obj: dict):
#         if "cands_selected" in json_obj:
#             cands_selected = json_obj["cands_selected"]
#             return cands_selected
#         else:
#             return None

# import json
# import re

# available_rules = [
#         "gomory", "pscost", "inference", "mostinf", "relpscost", "leastinf",
#         "distribution", "fullstrong", "cloud", "lookahead", "multaggr",
#         "allfullstrong", "vanillafullstrong", "random", "nodereopt", "multinode"
#     ]
# def parse_llm_output(output_str):
#     try:
#         parsed = json.loads(clean_llm_reply(output_str))
#         return parsed  # 返回 dict 类型
#     except json.JSONDecodeError as e:
#         print("JSON解析失败:", e)
#         return None

# def clean_llm_reply(reply: str):
#     # clean the generaton string
#     reply = re.sub(r"^```[a-zA-Z]*\n", "", reply.strip())
#     reply = re.sub(r"\n```$", "", reply.strip())
#     return reply
import json, re

available_rules = [
    "gomory", "pscost", "inference", "mostinf", "relpscost", "leastinf",
    "distribution", "fullstrong", "cloud", "lookahead", "multaggr",
    "allfullstrong", "vanillafullstrong", "random", "nodereopt", "multinode"
]
_AVAILABLE = set(available_rules)
_FALLBACK = "relpscost"

def _clean_llm_reply(reply: str) -> str:
    # 去掉 ```json ... ``` 包裹
    s = reply.strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _first_json_obj(s: str):
    """从任意文本中提取第一个 {...} JSON 并解析；失败返回 None"""
    for m in re.finditer(r"\{.*?\}", s, flags=re.S):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return None

def _normalize_rule(name: str) -> str:
    if not isinstance(name, str):
        return _FALLBACK
    r = name.strip().lower()
    return r if r in _AVAILABLE else _FALLBACK

def parse_llm_output(output_str):
    """
    返回两种规范化形式之一（根据输入决定）：
    - {"branching_rule": "<rule>"} 或
    - {"switch": <bool>, "next_rule": "<rule>"}
    若无法解析/不在可用列表，回退 relpscost。
    """
    text = _clean_llm_reply(output_str)

    obj = None
    # 先直接整体 JSON
    try:
        obj = json.loads(text)
    except Exception:
        # 再尝试从文本中抓第一个 {...}
        obj = _first_json_obj(text)

    if not isinstance(obj, dict):
        # 彻底解析失败 → 默认
        return {"branching_rule": _FALLBACK}

    # 方案 A：单键形式
    if "branching_rule" in obj:
        rule = _normalize_rule(obj.get("branching_rule"))
        return {"branching_rule": rule}

    # 方案 B：切换决策形式
    if "next_rule" in obj or "switch" in obj:
        rule = _normalize_rule(obj.get("next_rule"))
        sw = bool(obj.get("switch", False))
        return {"switch": sw, "next_rule": rule}

    # 其它意外结构 → 默认
    return {"branching_rule": _FALLBACK}



def parse_llm_reply(llm_reply: str):
    ok, val = extract_json_from_text_string(llm_reply)
    if not ok:
        return False, None
    # 直接返回字符串或列表（取第一个）
    if isinstance(val, str):
        return True, val
    if isinstance(val, list) and isinstance(val[0], str):
        return True, val[0]
    return False, None


def extract_json_from_text_string(text_str: str):
    # 优先直接尝试整体 JSON
    try:
        obj = json.loads(text_str)
        v = get_cands_selected(obj)
        if v is not None:
            return True, v
    except Exception:
        pass
    # 退化：在大括号片段中找
    for js in re.findall(r'\{.*?\}', text_str, flags=re.S):
        try:
            obj = json.loads(js)
            v = get_cands_selected(obj)
            if v is not None:
                return True, v
        except Exception:
            continue
    return False, None

def get_cands_selected(obj: dict):
    if "branching_rule" in obj:
        return obj["branching_rule"]
    return None
