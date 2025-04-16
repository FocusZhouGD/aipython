import requests
import json


connection = requests.get("http://localhost:11434")

print(f"连接状态: {connection}")
#print(connection.json().get("models"))


response = requests.get("http://localhost:11434/api/tags")

print(f"模型tags: {response.json().get('models')}")
model_tags = response.json().get('models',[])

#

model_names = [x["name"] for x in model_tags]

print(f"模型名称: {model_names}")


def chat_completion(base_url:str,model_name:str):
    """
    测试 聊天
    
    """
    url = f"{base_url}/api/chat"
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是一个AI助手，请根据用户的问题给出回答。"},
            {"role": "user", "content": "你好"}
        ],
        "stream": False
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"API调用成功！")
        print(f"没有json: {result}")
        print(f"API响应: {json.dumps(result,ensure_ascii=False)}")
    else:
        print(f"API调用失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")


print(chat_completion("http://localhost:11434","deepseek-r1:7b"))


def text_completion(base_url:str,model_name:str,prompt:str):
    """
    测试 文本生成
    """
    url = f"{base_url}/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"API调用成功！")
        print(f"API文本生成响应: {json.dumps(result,ensure_ascii=False)}")
        #提取文本
        if "response" in result:
            text = result["response"]
            print(f"文本生成结果: {text}")
        else:
            print(f"没有提取到文本")
    else:
        print(f"API调用失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")     


print(text_completion("http://localhost:11434","deepseek-r1:7b","请写一首关于夏天的诗"))


def embedding(base_url:str,model_name:str,text:str):
    """
    测试 文本嵌入
    """
    url = f"{base_url}/api/embeddings"
    data = {
        "model": model_name,
        "prompt": text
      
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"API调用成功！")
        print(f"API文本嵌入响应: {json.dumps(result,ensure_ascii=False)}")
        if "embedding" in result:
            embedding = result["embedding"]
            print(f"文本嵌入结果: {len(embedding)}")
            print(f"文本嵌入结果前5个: {embedding[:5]}")
        else:
            print(f"没有提取到文本嵌入")
    else:
        print(f"API调用失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")


print(embedding("http://localhost:11434","deepseek-r1:7b","这是一个示例文本"))


#if __name__ == "__main__":
