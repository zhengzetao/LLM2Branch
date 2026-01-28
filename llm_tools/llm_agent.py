import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

class OPLLMAgent:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def compose_messages(self, roles, msg_content_list):
        if len(roles) != len(msg_content_list):
            raise ValueError("角色列表和消息内容列表的长度不匹配")
        messages = [{"role": roles[i], "content": msg_content_list[i]} for i in range(len(roles))]
        return messages

    def get_reply(self, messages, model="qwen-turbo", temperature=0.7, max_tokens=8000):
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            response.encoding = 'utf-8'  # 显式设置编码
            result = response.json()
            reply = result['choices'][0]['message']['content']
            return reply

        except Exception as e:
            print(f"Error: {e}")
            return "Failed to get LLM reply"


# if __name__ == "__main__":
#     client = OPLLMAgent(
#         api_key="sk-3c1c5f5a8d774277b16292772599406a",
#         base_url="https://dashscope.aliyuncs.com"
#     )
#     prompt = "你是谁"
#     messages = client.compose_messages(roles=['user'], msg_content_list=[prompt])
#     reply = client.get_reply(
#         messages=messages,
#         model="qwen-turbo",
#         temperature=0.7,
#         max_tokens=8000
#     )

#     print(reply)


# class llmClient():

#     def __init__(self,
#                  api_key,
#                  model_name="Qwen/Qwen3-30B-A3B",
#                  api_url="https://api.siliconflow.cn/v1/chat/completions",
#                  stream=False,
#                  max_tokens=4096,
#                  enable_thinking=False,
#                  thinking_budget=4096,
#                  min_p=0.05,
#                  temperature=0.5,
#                  top_p=0.7,
#                  top_k=50,
#                  frequency_penalty=0.8,
#                  n=1
#                  ):

#         self.api_key = api_key
#         self.model_name = model_name
#         self.api_url = api_url
#         self.stream = stream
#         self.max_tokens = max_tokens
#         self.enable_thinking = enable_thinking
#         self.thinking_budget = thinking_budget
#         self.min_p = min_p
#         self.temperature = temperature
#         self.top_p = top_p
#         self.top_k = top_k
#         self.frequency_penalty = frequency_penalty
#         self.n = n

#     def getResponse(self, prompt):

#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }

#         payload = {
#             "model": self.model_name,
#             "stream": self.stream,
#             "max_tokens": self.max_tokens,
#             "enable_thinking": self.enable_thinking,
#             "thinking_budget": self.thinking_budget,
#             "min_p": self.min_p,
#             "temperature": self.temperature,
#             "top_p": self.top_p,
#             "top_k": self.top_k,
#             "frequency_penalty": self.frequency_penalty,
#             "n": self.n,
#             "stop": [],
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ]
#         }

#         try:
#             response = requests.post(self.api_url, json=payload, headers=headers)
#             response.raise_for_status()  # 自动触发HTTP错误异常
#             return response.text
#         except Exception as e:
#             print(f"API请求失败: {str(e)}")
#             return None

#     def get_content(self, response_text):
#         if not response_text or not isinstance(response_text, str):
#             print(f"无效响应类型: {type(response_text)}")
#             return None

#         try:
#             response_json = json.loads(response_text)

#             # 层级校验
#             if 'choices' not in response_json:
#                 print(f"响应缺少choices字段，完整响应：{response_text}")
#                 return None

#             if not isinstance(response_json['choices'], list) or len(response_json['choices']) == 0:
#                 print(f"choices格式异常，完整响应：{response_text}")
#                 return None

#             message = response_json['choices'][0].get('message', {})
#             return message.get('content')

#         except json.JSONDecodeError:
#             print(f"响应非JSON格式，原始内容：{response_text}")
#             return None
#         except Exception as e:
#             print(f"解析异常: {str(e)}")
#             return None

#     def safe_get_content(self, prompt):
#         raw_response = self.getResponse(prompt)
#         if raw_response is None:
#             return "请求失败，请检查网络和API配置"
#         return self.get_content(raw_response) or "内容解析失败"


# 可放在模块级
AVAILABLE_RULES = {
    "pscost","relpscost","mostinf","leastinf","fullstrong",
    "allfullstrong","vanillafullstrong","inference","distribution",
    "cloud","lookahead","multaggr","random","nodereopt","multinode"
}
FALLBACK_RULE = "pscost"

def _split_models(model_name: str):
    if isinstance(model_name, str):
        # 允许 "a, b , c"
        models = [m.strip() for m in model_name.split(",") if m.strip()]
        return models if models else [model_name.strip()]
    elif isinstance(model_name, (list, tuple)):
        return [str(x).strip() for x in model_name if str(x).strip()]
    return [str(model_name).strip()]

def _normalize_rule(name: str) -> str:
    if not isinstance(name, str):
        return None
    r = name.strip().lower()
    return r if r in AVAILABLE_RULES else None


class llmClient:
    def __init__(self,
                 api_key,
                 model_name="Qwen/Qwen3-30B-A3B",
                 api_url="https://api.siliconflow.cn/v1/chat/completions",
                 stream=False,
                 max_tokens=4096,
                 enable_thinking=False,
                 thinking_budget=4096,
                 min_p=0.05,
                 temperature=0.5,
                 top_p=0.7,
                 top_k=50,
                 frequency_penalty=0.8,
                 n=1):
        self.api_key = api_key
        self.api_url = api_url
        self.stream = stream
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.min_p = min_p
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.frequency_penalty = frequency_penalty
        self.n = n

        # 支持逗号分隔的多模型
        self.models = _split_models(model_name)

    # 可选：动态设置模型
    def set_models(self, model_name_or_list):
        self.models = _split_models(model_name_or_list)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _payload(self, prompt, model):
        return {
            "model": model,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
            "thinking_budget": self.thinking_budget,
            "min_p": self.min_p,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "n": self.n,
            "stop": [],
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

    def getResponse(self, prompt, model=None):
        model = model or (self.models[0] if self.models else None)
        try:
            resp = requests.post(self.api_url,
                                 json=self._payload(prompt, model),
                                 headers=self._headers(),
                                 timeout=60)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"[LLM] request failed for model={model}: {e}")
            return None

    def get_content(self, response_text):
        if not response_text or not isinstance(response_text, str):
            return None
        try:
            data = json.loads(response_text)
            ch = data.get("choices")
            if not ch or not isinstance(ch, list):
                return None
            msg = ch[0].get("message", {})
            return msg.get("content")
        except Exception:
            return None

    def safe_get_content(self, prompt, model=None):
        raw = self.getResponse(prompt, model=model)
        return self.get_content(raw)

    # ====== 新增：单/多模型自适应推断并投票 ======
    def infer_rule(self, prompt, parse_fn, fallback=FALLBACK_RULE, parallel=True):
        """
        parse_fn: 函数，将 content(str) -> dict
                  支持 {"branching_rule": "..."} 或 {"switch": bool, "next_rule": "..."}
        返回: {"branching_rule": "<rule>"}
        """
        models = self.models or []
        if not models:
            print('without LLM models')
            return {"branching_rule": fallback}

        # 单模型：直接跑一次
        if len(models) == 1 or not parallel:
            content = self.safe_get_content(prompt, model=models[0])
            parsed = parse_fn(content) if content else None
            rule = None
            if isinstance(parsed, dict):
                if "branching_rule" in parsed:
                    rule = _normalize_rule(parsed.get("branching_rule"))
                elif "next_rule" in parsed:
                    rule = _normalize_rule(parsed.get("next_rule"))
            print('Only one model!')
            return {"branching_rule": rule or fallback}

        # 多模型并行：投票
        votes = []
        def _worker(m):
            content = self.safe_get_content(prompt, model=m)
            parsed = parse_fn(content) if content else None
            if isinstance(parsed, dict):
                if "branching_rule" in parsed:
                    r = _normalize_rule(parsed.get("branching_rule"))
                else:
                    r = _normalize_rule(parsed.get("next_rule"))
                return r
            return None

        with ThreadPoolExecutor(max_workers=min(8, len(models))) as ex:
            futs = {ex.submit(_worker, m): m for m in models}
            for fut in as_completed(futs):
                try:
                    r = fut.result()
                    if r: votes.append(r)
                except Exception as e:
                    print(f"[LLM] model={futs[fut]} failed: {e}")

        if not votes:
            print('NO rule recommended, using default setting')
            return {"branching_rule": fallback}

        cnt = Counter(votes)
        print('voting results:', cnt)
        # 平票按 models 顺序优先
        best, _ = max(cnt.items(), key=lambda kv: (kv[1], -min(models.index(m) for m in models if m and kv[0])))
        return {"branching_rule": best}
