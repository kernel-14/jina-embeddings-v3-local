# 部署流程
本项目使用 FastAPI 框架在本地部署 jinaai/jina-embeddings-v3 模型，并提供一个与 Jina 官方 API 格式兼容的 POST /embd 接口。

服务器代码: server_main.py

测试脚本 : test_embd.py

## 必要包安装

### 创建 Conda 环境
```bash
# 创建一个环境 jina_project
conda create -n jina_project python=3.10
conda activate jina_project

```

### 安装PyTorch

```Bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 安装其他Python依赖

这些是 server_main.py 和 test_embd.py 运行所必需的包。

```Bash

pip install fastapi uvicorn sentence-transformers einops huggingface-hub regex

```

fastapi: Web 框架。

uvicorn: ASGI 服务器，用于启动 fastapi。

sentence-transformers: 模型加载器。

einops: jina-embeddings-v3 模型的必需依赖。

huggingface-hub: 用于登录 Hugging Face。

regex: test_embd.py 脚本中的 trim_symbols 函数需要。

## 部署流程

### 启动 API 服务器
在终端运行 uvicorn 来启动 server_main.py：

```Bash
uvicorn server_main:app --host 0.0.0.0 --port 8000
```
请保持此终端不要关闭。

成功启动后，你将看到 Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)。

（首次启动会自动下载模型，后续启动会非常快。）

服务器启动后，打开浏览器访问 http://127.0.0.1:8000/docs 即可看到 FastAPI 自动生成的交互式 API 文档。

### 运行测试脚本
打开一个新的终端，激活同一个 conda 环境，运行 test_embd.py来测试：

```Bash

python test_embd.py
```

此脚本会调用 http://127.0.0.1:8000/embd 并打印出服务器返回的原始 JSON。

你会看到类似 http://127.0.0.1:8000 "POST /embd HTTP/1.1" 200 OK 的成功日志。

同时，终端 1（服务器）也会打印出 INFO: 127.0.0.1:..... - "POST /embd HTTP/1.1" 200 OK 的访问记录。

# API文档

## 1. 接口概述

- 接口描述: `Jina-embeddings-v3` 模型本地 API 接口，用于将文本列表转换为向量列表。
- 请求路径: `http://127.0.0.1:8000/embd`
- 请求方法: `POST`

## 2. 请求 (Request)

- 请求格式: `application/json`
- 请求参数说明:  (基于 ApiRequest 类)

| **参数名** | **类型** | **默认值** | **参数说明** |
| --- | --- | --- | --- |
| `input` | `array (string)` | `N/A` | 需要被转换成向量的字符串列表 |
| `model` | `string` | `None` | 模型名称 `jinaai/jina-embeddings-v3` |
| `task` | `string` | `None` | 任务类型 |
| `dimensions` | `integer` | `None` |  输出维度 |
| `embedding_type` | `string` | `None` |  Embedding 类型 |
| `truncate` | `boolean` | `None` |  是否截断 |
| `late_chunking` | `boolean` | `None` | 是否晚分块 |
- 请求样例:JSON
    
    ```
    data_to_send = {
            "model": "jina-embeddings-v3",
            "task": "text-matching",
            "input": [
                "你好，我是测试客户端",
                "我正在调用你本地的服务器"
            ]
        }
    ```
    

## 3. 响应 (Response)

- 响应格式: `application/json`
- 响应参数说明:

| **参数名** | **类型** | **参数说明** |
| --- | --- | --- |
| `model` | `string` | 所使用的模型名称 (e.g., "jinaai/jina-embeddings-v3")。 |
| `object` | `string` | 响应类型，恒定为 `"list"`。 |
| `usage` | `object` | 包含 Token 数量的对象。 |
| `usage.total_tokens` | `integer` | 本次请求消耗的总 Token 数。 |
| `usage.prompt_tokens` | `integer` | 本次请求的提示 Token 数 (同 `total_tokens`)。 |
| `data` | `array (object)` | 包含 embedding 结果的列表。 |
| `data[].object` | `string` | 数据类型，恒定为 `"embedding"`。 |
| `data[].index` | `integer` | 向量在原始 `input` 列表中的索引号。 |
| `data[].embedding` | `array (number)` | 浮点数向量本身。 |
- 响应样例:JSON
    
    ![image.png](image.png)
    

```json
{
  "model": "jinaai/jina-embeddings-v3",
  "object": "list",
  "usage": {
    "total_tokens": int,
    "prompt_tokens": int
  },
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        ...
      ]
    },
    {
      "object": "embedding",
      "index": 1,
      "embedding": [
        ...
      ]
    }
  ],
}
```

