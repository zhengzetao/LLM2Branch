import os
import requests
import json
import pdb

class OPLLMAgent:
    def __init__(self, api_key=None, base_url=None):
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
        self.api_key = api_key
        self.base_url = base_url

    def compose_messages(self, roles, msg_content_list):
        if len(roles) != len(msg_content_list):
            raise ValueError("角色列表和消息内容列表的长度不匹配")
        messages = [{"role": roles[i], "content": msg_content_list[i]} for i in range(len(roles))]
        return messages

    def get_reply(self, messages, model="qwen-turbo-latest", temperature=0.7, max_tokens=8000):
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
        json_data = json.dumps(data, ensure_ascii=False)

        try:
            response = requests.post(url, headers=headers, data=json_data.encode('utf-8'))
            response.raise_for_status()
            response.encoding = 'utf-8'  # 显式设置编码
            result = response.json()
            choice = result['choices'][0]
            reply = choice['message']['content']
            return reply

        except Exception as e:
            print(f"Error: {e}")
            return "Failed to get LLM reply"