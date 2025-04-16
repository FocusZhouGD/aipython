import os
import requests
import json
from typing import List, Dict, Any, Optional, Union


class OllamaClient:
    """
    Ollama API客户端类，用于与Ollama服务交互
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:11434",
        api_key: Optional[str] = None
    ):
        """
        初始化Ollama客户端
        
        Args:
            base_url: Ollama API的基础URL，默认为http://localhost:11434
            api_key: API密钥，Ollama通常不需要API密钥，但可以设置
        """
        self.base_url = base_url
        self.api_key = api_key
        
        # 设置请求头
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
            
        print(f"使用Ollama API，基础URL: {self.base_url}")
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "llama2",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用Ollama聊天完成API
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "你好"}, ...]
            model: 使用的模型名称，默认为llama2
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成的token数量
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        url = f"{self.base_url}/api/chat"
        
        # 构建请求参数
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        # 添加选项参数
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options
            
        # 添加其他参数
        for key, value in kwargs.items():
            if key not in ["model", "messages", "stream", "options"]:
                if "options" not in payload:
                    payload["options"] = {}
                payload["options"][key] = value
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"调用Ollama聊天完成API时出错: {e}")
            raise
    
    def get_completion_text(self, response: Dict[str, Any]) -> str:
        """
        从聊天完成API响应中提取生成的文本
        
        Args:
            response: API响应结果
            
        Returns:
            生成的文本内容
        """
        try:
            if "message" in response:
                return response["message"]["content"]
            else:
                print(f"无法从响应中提取文本: {response}")
                return ""
        except (KeyError, IndexError, AttributeError) as e:
            print(f"从响应中提取文本时出错: {e}")
            return ""
    
    def text_completion(
        self, 
        prompt: str, 
        model: str = "llama2",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用Ollama文本生成API
        
        Args:
            prompt: 提示文本
            model: 使用的模型名称，默认为llama2
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成的token数量
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        url = f"{self.base_url}/api/generate"
        
        # 构建请求参数
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # 添加选项参数
        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            payload["options"] = options
            
        # 添加其他参数
        for key, value in kwargs.items():
            if key not in ["model", "prompt", "stream", "options"]:
                if "options" not in payload:
                    payload["options"] = {}
                payload["options"][key] = value
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"调用Ollama文本生成API时出错: {e}")
            raise
    
    def get_text_completion(self, response: Dict[str, Any]) -> str:
        """
        从文本生成API响应中提取生成的文本
        
        Args:
            response: API响应结果
            
        Returns:
            生成的文本内容
        """
        try:
            if "response" in response:
                return response["response"]
            else:
                print(f"无法从响应中提取文本: {response}")
                return ""
        except (KeyError, IndexError, AttributeError) as e:
            print(f"从响应中提取文本时出错: {e}")
            return ""
    
    def create_embedding(
        self, 
        text: str, 
        model: str = "llama2"
    ) -> List[float]:
        """
        创建文本的嵌入向量
        
        Args:
            text: 要创建嵌入的文本
            model: 使用的模型名称，默认为llama2
            
        Returns:
            嵌入向量
        """
        url = f"{self.base_url}/api/embeddings"
        
        payload = {
            "model": model,
            "prompt": text
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            print(f"创建嵌入向量时出错: {e}")
            raise
    
    def list_models(self) -> List[str]:
        """
        获取Ollama可用的模型列表
        
        Returns:
            可用模型名称列表
        """
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except Exception as e:
            print(f"获取Ollama模型列表时出错: {e}")
            return []
    
    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """
        从Ollama服务器拉取模型
        
        Args:
            model_name: 要拉取的模型名称
            
        Returns:
            API响应结果
        """
        url = f"{self.base_url}/api/pull"
        
        payload = {
            "name": model_name
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"拉取模型时出错: {e}")
            raise


# 使用示例
if __name__ == "__main__":
    # 初始化Ollama客户端
    client = OllamaClient(base_url="http://:11434")
    
    # 获取可用模型列表
    models = client.list_models()
    print("可用Ollama模型:", models)
    
    if models:
        # 使用第一个可用模型
        model = models[2]
        print(f"使用模型: {model}")
        
        # 聊天完成示例
        messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ]
        
        print("发送聊天请求到Ollama API...")
        response = client.chat_completion(messages, model=model)
        print("Ollama聊天回复:", client.get_completion_text(response))
        
        # 文本生成示例
        prompt = "请写一首关于春天的诗"
        print("\n发送文本生成请求到Ollama API...")
        response = client.text_completion(prompt, model=model)
        print("Ollama文本生成:", client.get_text_completion(response))
        
        # 嵌入向量示例
        text = "这是一个示例文本"
        print("\n发送嵌入向量请求到Ollama API...")
        embedding = client.create_embedding(text, model=model)
        print("嵌入向量维度:", len(embedding))
    else:
        print("未找到可用模型，请确保Ollama服务已启动并安装了模型")