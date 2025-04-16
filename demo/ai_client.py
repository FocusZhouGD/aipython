import os
import requests
import json
from typing import List, Dict, Any, Optional, Union
from enum import Enum


class AIProvider(Enum):
    """AI服务提供商枚举"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    LOCAL = "local"


class AIClient:
    """
    通用AI客户端类，支持多种AI服务提供商
    """
    
    def __init__(
        self, 
        provider: Union[str, AIProvider] = AIProvider.OPENAI,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_local_fallback: bool = True
    ):
        """
        初始化AI客户端
        
        Args:
            provider: AI服务提供商，可选值: "openai", "deepseek", "ollama", "local"
            api_key: API密钥，如果为None则尝试从环境变量获取
            base_url: API基础URL，如果为None则使用默认值
            use_local_fallback: 是否在API不可用时使用本地回退选项
        """
        self.provider = provider if isinstance(provider, AIProvider) else AIProvider(provider)
        self.use_local_fallback = use_local_fallback
        
        # 设置API密钥
        if self.provider == AIProvider.OPENAI:
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key and not use_local_fallback:
                raise ValueError("必须提供OpenAI API密钥或设置OPENAI_API_KEY环境变量")
            self.base_url = base_url or "https://api.openai.com/v1"
            self.client = None
            if self.api_key:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(api_key=self.api_key)
                except ImportError:
                    print("警告: 未安装openai库，将使用requests直接调用API")
                
        elif self.provider == AIProvider.DEEPSEEK:
            self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
            if not self.api_key and not use_local_fallback:
                raise ValueError("必须提供DeepSeek API密钥或设置DEEPSEEK_API_KEY环境变量")
            self.base_url = base_url or "https://api.deepseek.com/v1"
            
        elif self.provider == AIProvider.OLLAMA:
            # Ollama通常不需要API密钥，但可以设置
            self.api_key = api_key
            self.base_url = base_url or "http://localhost:11434"
            print(f"使用Ollama API，基础URL: {self.base_url}")
            
        else:  # LOCAL
            self.api_key = None
            self.base_url = None
            print("使用本地模型模式，需要确保已安装相关依赖")
        
        # 设置请求头
        self.headers = {}
        if self.api_key:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        else:
            self.headers = {"Content-Type": "application/json"}
    
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用聊天完成API
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "你好"}, ...]
            model: 使用的模型名称，如果为None则使用默认模型
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成的token数量
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        # 设置默认模型
        if model is None:
            if self.provider == AIProvider.OPENAI:
                model = "gpt-3.5-turbo"
            elif self.provider == AIProvider.DEEPSEEK:
                model = "deepseek-chat"
            elif self.provider == AIProvider.OLLAMA:
                model = "llama2"  # Ollama默认模型
            else:
                model = "local-model"
        
        try:
            if self.provider == AIProvider.OPENAI and self.client:
                # 使用OpenAI客户端库
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response
                
            elif self.provider == AIProvider.OPENAI:
                # 使用requests直接调用OpenAI API
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
                response.raise_for_status()
                return response.json()
                
            elif self.provider == AIProvider.DEEPSEEK:
                # 调用DeepSeek API
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
                response.raise_for_status()
                return response.json()
                
            elif self.provider == AIProvider.OLLAMA:
                # 调用Ollama API
                url = f"{self.base_url}/api/chat"
                
                # Ollama的API格式略有不同
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
                
                response = requests.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()
                
            else:  # LOCAL
                # 本地模型回退选项
                return self._local_chat_completion(messages, model, temperature, max_tokens, **kwargs)
                
        except Exception as e:
            print(f"调用{self.provider.value} API时出错: {e}")
            if self.use_local_fallback and self.provider != AIProvider.LOCAL:
                print("尝试使用本地回退选项...")
                return self._local_chat_completion(messages, model, temperature, max_tokens, **kwargs)
            raise
    
    def _local_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        本地聊天完成实现（回退选项）
        
        这里可以实现一个简单的本地模型调用，或者返回一个模拟响应
        """
        print("使用本地模型回退选项")
        
        # 这里只是一个简单的模拟实现
        # 在实际应用中，您可以集成本地模型，如通过Hugging Face的transformers库
        
        # 模拟响应
        return {
            "choices": [
                {
                    "message": {
                        "content": "这是一个本地模型生成的回复。在实际应用中，您可以集成真正的本地模型。",
                        "role": "assistant"
                    },
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "model": model,
            "usage": {
                "completion_tokens": 20,
                "prompt_tokens": 10,
                "total_tokens": 30
            }
        }
    
    def get_completion_text(self, response: Dict[str, Any]) -> str:
        """
        从API响应中提取生成的文本
        
        Args:
            response: API响应结果
            
        Returns:
            生成的文本内容
        """
        try:
            if self.provider == AIProvider.OPENAI and hasattr(response, 'choices'):
                # OpenAI客户端库响应
                return response.choices[0].message.content
            elif self.provider == AIProvider.OLLAMA and "message" in response:
                # Ollama API响应
                return response["message"]["content"]
            elif isinstance(response, dict) and "choices" in response:
                # 标准API响应格式
                return response["choices"][0]["message"]["content"]
            else:
                print(f"无法从响应中提取文本: {response}")
                return ""
        except (KeyError, IndexError, AttributeError) as e:
            print(f"从响应中提取文本时出错: {e}")
            return ""
    
    def text_completion(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        调用文本完成API
        
        Args:
            prompt: 提示文本
            model: 使用的模型名称，如果为None则使用默认模型
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成的token数量
            **kwargs: 其他参数
            
        Returns:
            API响应结果
        """
        # 设置默认模型
        if model is None:
            if self.provider == AIProvider.OPENAI:
                model = "gpt-3.5-turbo-instruct"
            elif self.provider == AIProvider.DEEPSEEK:
                model = "deepseek-coder"
            elif self.provider == AIProvider.OLLAMA:
                model = "llama2"  # Ollama默认模型
            else:
                model = "local-model"
        
        try:
            if self.provider == AIProvider.OPENAI and self.client:
                # 使用OpenAI客户端库
                response = self.client.completions.create(
                    model=model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return response
                
            elif self.provider == AIProvider.OPENAI:
                # 使用requests直接调用OpenAI API
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
                response.raise_for_status()
                return response.json()
                
            elif self.provider == AIProvider.DEEPSEEK:
                # 调用DeepSeek API
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
                response.raise_for_status()
                return response.json()
                
            elif self.provider == AIProvider.OLLAMA:
                # 调用Ollama API
                url = f"{self.base_url}/api/generate"
                
                # Ollama的API格式略有不同
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
                
                response = requests.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()
                
            else:  # LOCAL
                # 本地模型回退选项
                return self._local_text_completion(prompt, model, temperature, max_tokens, **kwargs)
                
        except Exception as e:
            print(f"调用{self.provider.value} API时出错: {e}")
            if self.use_local_fallback and self.provider != AIProvider.LOCAL:
                print("尝试使用本地回退选项...")
                return self._local_text_completion(prompt, model, temperature, max_tokens, **kwargs)
            raise
    
    def _local_text_completion(
        self, 
        prompt: str, 
        model: str = "local-model",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        本地文本完成实现（回退选项）
        """
        print("使用本地模型回退选项")
        
        # 模拟响应
        return {
            "choices": [
                {
                    "text": "这是一个本地模型生成的文本完成。在实际应用中，您可以集成真正的本地模型。",
                    "finish_reason": "stop",
                    "index": 0
                }
            ],
            "model": model,
            "usage": {
                "completion_tokens": 20,
                "prompt_tokens": 10,
                "total_tokens": 30
            }
        }
    
    def get_text_completion(self, response: Dict[str, Any]) -> str:
        """
        从文本完成API响应中提取生成的文本
        
        Args:
            response: API响应结果
            
        Returns:
            生成的文本内容
        """
        try:
            if self.provider == AIProvider.OPENAI and hasattr(response, 'choices'):
                # OpenAI客户端库响应
                return response.choices[0].text
            elif self.provider == AIProvider.OLLAMA and "response" in response:
                # Ollama API响应
                return response["response"]
            elif isinstance(response, dict) and "choices" in response:
                # 标准API响应格式
                return response["choices"][0]["text"]
            else:
                print(f"无法从响应中提取文本: {response}")
                return ""
        except (KeyError, IndexError, AttributeError) as e:
            print(f"从响应中提取文本时出错: {e}")
            return ""
    
    def create_embedding(
        self, 
        text: str, 
        model: Optional[str] = None
    ) -> List[float]:
        """
        创建文本的嵌入向量
        
        Args:
            text: 要创建嵌入的文本
            model: 使用的模型名称，如果为None则使用默认模型
            
        Returns:
            嵌入向量
        """
        # 设置默认模型
        if model is None:
            if self.provider == AIProvider.OPENAI:
                model = "text-embedding-ada-002"
            elif self.provider == AIProvider.DEEPSEEK:
                model = "deepseek-embedding"
            elif self.provider == AIProvider.OLLAMA:
                model = "llama2"  # Ollama默认模型
            else:
                model = "local-embedding"
        
        try:
            if self.provider == AIProvider.OPENAI and self.client:
                # 使用OpenAI客户端库
                response = self.client.embeddings.create(
                    model=model,
                    input=text
                )
                return response.data[0].embedding
                
            elif self.provider == AIProvider.OPENAI:
                # 使用requests直接调用OpenAI API
                url = f"{self.base_url}/embeddings"
                payload = {
                    "model": model,
                    "input": text
                }
                    
                response = requests.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
                
            elif self.provider == AIProvider.DEEPSEEK:
                # 调用DeepSeek API
                url = f"{self.base_url}/embeddings"
                payload = {
                    "model": model,
                    "input": text
                }
                    
                response = requests.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()["data"][0]["embedding"]
                
            elif self.provider == AIProvider.OLLAMA:
                # 调用Ollama API
                url = f"{self.base_url}/api/embeddings"
                
                payload = {
                    "model": model,
                    "prompt": text
                }
                
                response = requests.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()["embedding"]
                
            else:  # LOCAL
                # 本地模型回退选项
                return self._local_embedding(text, model)
                
        except Exception as e:
            print(f"创建嵌入向量时出错: {e}")
            if self.use_local_fallback and self.provider != AIProvider.LOCAL:
                print("尝试使用本地回退选项...")
                return self._local_embedding(text, model)
            raise
    
    def _local_embedding(
        self, 
        text: str, 
        model: str = "local-embedding"
    ) -> List[float]:
        """
        本地嵌入向量实现（回退选项）
        """
        print("使用本地模型回退选项")
        
        # 返回一个简单的随机向量作为模拟嵌入
        import random
        return [random.random() for _ in range(1536)]  # 1536是常见的嵌入维度
    
    def list_ollama_models(self) -> List[str]:
        """
        获取Ollama可用的模型列表
        
        Returns:
            可用模型名称列表
        """
        if self.provider != AIProvider.OLLAMA:
            print("此方法仅适用于Ollama提供商")
            return []
            
        try:
            url = f"{self.base_url}/api/tags"
            response = requests.get(url)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except Exception as e:
            print(f"获取Ollama模型列表时出错: {e}")
            return []


# 使用示例
if __name__ == "__main__":
    # 尝试使用OpenAI
    try:
        client = AIClient(provider=AIProvider.OPENAI, use_local_fallback=True)
        
        # 聊天完成示例
        messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ]
        
        response = client.chat_completion(messages)
        print("聊天回复:", client.get_completion_text(response))
        
    except Exception as e:
        print(f"OpenAI API调用失败: {e}")
    
    # 尝试使用DeepSeek
    try:
        client = AIClient(provider=AIProvider.DEEPSEEK, use_local_fallback=True)
        
        # 聊天完成示例
        messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ]
        
        response = client.chat_completion(messages)
        print("聊天回复:", client.get_completion_text(response))
        
    except Exception as e:
        print(f"DeepSeek API调用失败: {e}")
    
    # 尝试使用Ollama
    try:
        # 使用自定义Ollama API地址
        client = AIClient(
            provider=AIProvider.OLLAMA, 
            base_url="http://:11434",  # 替换为您的Ollama API地址
            use_local_fallback=True
        )
        
        # 获取可用模型列表
        models = client.list_ollama_models()
        print("可用Ollama模型:", models)
        
        # 聊天完成示例
        messages = [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ]
        
        # 使用第一个可用模型，如果没有则使用默认模型
        model = models[2] if models else "llama2"
        print(f"使用模型: {model}")
        
        # 打印请求详情以便调试
        print("发送请求到Ollama API...")
        response = client.chat_completion(messages, model=model)
        print("Ollama聊天回复:", client.get_completion_text(response))
        
    except Exception as e:
        print(f"Ollama API调用失败: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()
    
    # 使用本地模型
    client = AIClient(provider=AIProvider.LOCAL)
    
    # 聊天完成示例
    messages = [
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]
    
    response = client.chat_completion(messages)
    print("本地模型回复:", client.get_completion_text(response)) 