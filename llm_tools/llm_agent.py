import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from llm_tools.rules import (
    CLIENT_FALLBACK_RULE,
    DYNAMIC_BRANCHING_RULES,
    normalize_rule,
    split_models,
)


class llmClient:
    def __init__(
        self,
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
        n=1,
    ):
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
        self.models = split_models(model_name)

    def set_models(self, model_name_or_list):
        self.models = split_models(model_name_or_list)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
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
                {"role": "user", "content": prompt},
            ],
        }

    def getResponse(self, prompt, model=None):
        model = model or (self.models[0] if self.models else None)
        try:
            response = requests.post(
                self.api_url,
                json=self._payload(prompt, model),
                headers=self._headers(),
                timeout=60,
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            print(f"[LLM] request failed for model={model}: {exc}")
            return None

    def get_content(self, response_text):
        if not response_text or not isinstance(response_text, str):
            return None
        try:
            data = json.loads(response_text)
            choices = data.get("choices")
            if not choices or not isinstance(choices, list):
                return None
            message = choices[0].get("message", {})
            return message.get("content")
        except Exception:
            return None

    def safe_get_content(self, prompt, model=None):
        raw_response = self.getResponse(prompt, model=model)
        return self.get_content(raw_response)

    def infer_rule(self, prompt, parse_fn, fallback=CLIENT_FALLBACK_RULE, parallel=True, current_rule=None):
        """
        Infer a branching rule from one or more LLM models.

        parse_fn should return either {"branching_rule": "..."} or
        {"switch": bool, "next_rule": "..."}.
        """
        models = self.models or []
        if not models:
            print("without LLM models")
            return {"branching_rule": fallback}

        if len(models) == 1 or not parallel:
            content = self.safe_get_content(prompt, model=models[0])
            parsed = parse_fn(content) if content else None
            rule = self._rule_from_parsed(parsed, current_rule=current_rule)
            print("Only one model!")
            return {"branching_rule": rule or fallback}

        votes = []

        def worker(model):
            content = self.safe_get_content(prompt, model=model)
            parsed = parse_fn(content) if content else None
            return self._rule_from_parsed(parsed, current_rule=current_rule)

        with ThreadPoolExecutor(max_workers=min(8, len(models))) as executor:
            futures = {executor.submit(worker, model): model for model in models}
            for future in as_completed(futures):
                try:
                    rule = future.result()
                    if rule:
                        votes.append(rule)
                except Exception as exc:
                    print(f"[LLM] model={futures[future]} failed: {exc}")

        if not votes:
            print("NO rule recommended, using default setting")
            return {"branching_rule": fallback}

        counts = Counter(votes)
        print("voting results:", counts)
        best, _ = max(
            counts.items(),
            key=lambda item: (
                item[1],
                -min(index for index, model in enumerate(models) if model and item[0]),
            ),
        )
        return {"branching_rule": best}

    def _rule_from_parsed(self, parsed, current_rule=None):
        if not isinstance(parsed, dict):
            return None
        if "branching_rule" in parsed:
            return normalize_rule(
                parsed.get("branching_rule"),
                available_rules=DYNAMIC_BRANCHING_RULES,
                fallback=None,
            )
        if "next_rule" in parsed:
            if not bool(parsed.get("switch", False)) and current_rule:
                return normalize_rule(
                    current_rule,
                    available_rules=DYNAMIC_BRANCHING_RULES,
                    fallback=None,
                )
            return normalize_rule(
                parsed.get("next_rule"),
                available_rules=DYNAMIC_BRANCHING_RULES,
                fallback=None,
            )
        return None
