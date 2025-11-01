import torch 
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Dict, Any

app = FastAPI(
    title = "Jina-embeddings-v3 本地部署服务器"
)

device = 'cuda'
model_name = 'jinaai/jina-embeddings-v3'
model = SentenceTransformer(model_name ,trust_remote_code=True).to(device)
model.eval()  # 评估模式，不参与训练

# define request & response
class ApiRequest(BaseModel):   
    input: List[str]
    model: Optional[str] = None
    task: Optional[str] = None
    dimensions : Optional[int] = None
    embedding_type : Optional[str] = None
    truncate: Optional[bool] = None
    late_chunking: Optional[bool] = None

class EmbeddingObject(BaseModel):
    object: str
    index: int
    embedding: List[float]

class UsageObject(BaseModel):
    total_tokens: int
    prompt_tokens: int

class ApiResponse(BaseModel):
    model: str
    object: str
    usage: UsageObject
    data: List[EmbeddingObject]

# api接口
@app.post("/embd",response_model=ApiResponse)

async def get_embeddings(request:ApiRequest):
    texts_to_process = request.input
    # compute array
    embeddings_np = model.encode(texts_to_process)
    embeddings_list = embeddings_np.tolist()

    # construt data list
    data_list = []
    for i,vector in enumerate(embeddings_list):
        data_list.append(EmbeddingObject(
            object="embedding",
            index=i,
            embedding=vector
            ))

    total_tokens = sum(len(text) for text in texts_to_process)



    return {
        "model": model_name,
        "object": "list",
        "data": data_list,
        "usage": {
            "total_tokens": total_tokens,
            "prompt_tokens": total_tokens
        }
    }
