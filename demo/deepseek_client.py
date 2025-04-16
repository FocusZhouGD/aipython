import os
import requests
import json
from typing import List, Dict, Any, Optional, Union


class DeepSeekClient:
    """
    DeepSeek API客户端类，提供与DeepSeek API交互的功能
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.deepseek.com/v1"):
        """
        初始化DeepSeek客户端
        
        Args:
            api_key: DeepSeek API密钥，如果为None则尝试从环境变量DEEPSEEK_API_KEY获取
            base_url: API基础URL
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("必须提供DeepSeek API密钥或设置DEEPSEEK_API_KEY环境变量")
        
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用DeepSeek的聊天完成API
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "你好"}, ...]
            model: 使用的模型名称
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成的token数量
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        try:
            url = f"{self.base_url}/chat/completions"
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                **kwargs
            }
            
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
                
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()  # 如果请求失败，抛出异常
            
            return response.json()
        except Exception as e:
            print(f"调用DeepSeek API时出错: {e}")
            raise
    
    def get_completion_text(self, response: Dict[str, Any]) -> str:
        """
        从API响应中提取生成的文本
        
        Args:
            response: API响应结果
            
        Returns:
            生成的文本内容
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            print(f"从响应中提取文本时出错: {e}")
            return ""
    
    def text_completion(
        self, 
        prompt: str, 
        model: str = "deepseek-coder",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用DeepSeek的文本完成API
        
        Args:
            prompt: 提示文本
            model: 使用的模型名称
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成的token数量
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        try:
            url = f"{self.base_url}/completions"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                **kwargs
            }
            
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
                
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()  # 如果请求失败，抛出异常
            
            return response.json()
        except Exception as e:
            print(f"调用DeepSeek API时出错: {e}")
            raise
    
    def get_text_completion(self, response: Dict[str, Any]) -> str:
        """
        从文本完成API响应中提取生成的文本
        
        Args:
            response: API响应结果
            
        Returns:
            生成的文本内容
        """
        try:
            return response["choices"][0]["text"]
        except (KeyError, IndexError) as e:
            print(f"从响应中提取文本时出错: {e}")
            return ""
    
    def create_embedding(
        self, 
        text: str, 
        model: str = "deepseek-embedding"
    ) -> List[float]:
        """
        创建文本的嵌入向量
        
        Args:
            text: 要创建嵌入的文本
            model: 使用的模型名称
            
        Returns:
            嵌入向量
        """
        try:
            url = f"{self.base_url}/embeddings"
            
            payload = {
                "model": model,
                "input": text
            }
                
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()  # 如果请求失败，抛出异常
            
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            print(f"创建嵌入向量时出错: {e}")
            raise


# 使用示例
if __name__ == "__main__":
    # 请替换为您的API密钥或设置环境变量
    client = DeepSeekClient(api_key="your-deepseek-api-key-here")
    
    # 聊天完成示例
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]
    
    response = client.chat_completion(messages)
    print("聊天回复:", client.get_completion_text(response))
    
    # 文本完成示例
    prompt = "请写一首关于春天的诗："
    response = client.text_completion(prompt)
    print("文本完成:", client.get_text_completion(response))
    
    # 嵌入向量示例
    embedding = client.create_embedding("这是一个测试文本")
    print("嵌入向量维度:", len(embedding)) 