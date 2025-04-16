import requests
import json
import sys

def test_ollama_connection(base_url="http://localhost:11434"):
    """
    测试与Ollama服务器的连接
    
    Args:
        base_url: Ollama API的基础URL
    """
    print(f"测试与Ollama服务器的连接: {base_url}")
    
    # 测试1: 检查服务器是否运行
    try:
        # 尝试获取模型列表
        url = f"{base_url}/api/tags"
        print(f"请求URL: {url}")
        
        response = requests.get(url)
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("成功连接到Ollama服务器")
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            print(f"找到 {len(model_names)} 个模型: {model_names}")
            return True, model_names
        else:
            print(f"连接失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False, []
    except Exception as e:
        print(f"连接测试出错: {e}")
        return False, []

def test_chat_completion(base_url="http://localhost:11434", model="llama2"):
    """
    测试Ollama聊天完成API
    
    Args:
        base_url: Ollama API的基础URL
        model: 使用的模型名称
    """
    print(f"\n测试聊天完成API，使用模型: {model}")
    
    url = f"{base_url}/api/chat"
    print(f"请求URL: {url}")
    
    # 构建请求参数
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个有用的助手。"},
            {"role": "user", "content": "你好，请介绍一下自己。"}
        ],
        "stream": False
    }
    
    print(f"请求参数: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API调用成功")
            print(f"响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            # 提取生成的文本
            if "message" in result:
                print(f"生成的文本: {result['message']['content']}")
            else:
                print("无法从响应中提取文本")
                
            return True
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"API调用出错: {e}")
        return False

def test_text_completion(base_url="http://localhost:11434", model="llama2"):
    """
    测试Ollama文本生成API
    
    Args:
        base_url: Ollama API的基础URL
        model: 使用的模型名称
    """
    print(f"\n测试文本生成API，使用模型: {model}")
    
    url = f"{base_url}/api/generate"
    print(f"请求URL: {url}")
    
    # 构建请求参数
    payload = {
        "model": model,
        "prompt": "请写一首关于春天的诗",
        "stream": False
    }
    
    print(f"请求参数: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API调用成功")
            print(f"响应内容: {json.dumps(result, ensure_ascii=False, indent=2)}")
            
            # 提取生成的文本
            if "response" in result:
                print(f"生成的文本: {result['response']}")
            else:
                print("无法从响应中提取文本")
                
            return True
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"API调用出错: {e}")
        return False

def test_embeddings(base_url="http://localhost:11434", model="llama2"):
    """
    测试Ollama嵌入向量API
    
    Args:
        base_url: Ollama API的基础URL
        model: 使用的模型名称
    """
    print(f"\n测试嵌入向量API，使用模型: {model}")
    
    url = f"{base_url}/api/embeddings"
    print(f"请求URL: {url}")
    
    # 构建请求参数
    payload = {
        "model": model,
        "prompt": "这是一个示例文本"
    }
    
    print(f"请求参数: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        response = requests.post(url, json=payload)
        print(f"响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API调用成功")
            
            # 提取嵌入向量
            if "embedding" in result:
                embedding = result["embedding"]
                print(f"嵌入向量维度: {len(embedding)}")
                print(f"嵌入向量前5个元素: {embedding[:5]}")
            else:
                print("无法从响应中提取嵌入向量")
                
            return True
        else:
            print(f"API调用失败，状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
    except Exception as e:
        print(f"API调用出错: {e}")
        return False

if __name__ == "__main__":
    # 获取命令行参数
    base_url = "http://localhost:11434"
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    
    print(f"使用Ollama API基础URL: {base_url}")
    
    # 测试连接
    success, models = test_ollama_connection(base_url)
    
    if success and models:
        # 使用第一个可用模型
        model = models[2]
        print(f"\n使用模型: {model}")
        
        # 测试聊天完成API
        test_chat_completion(base_url, model)
        
        # 测试文本生成API
        test_text_completion(base_url, model)
        
        # 测试嵌入向量API
        test_embeddings(base_url, model)
    else:
        print("\n无法连接到Ollama服务器或未找到可用模型")
        print("请确保:")
        print("1. Ollama服务已启动")
        print("2. API地址正确")
        print("3. 已安装至少一个模型")