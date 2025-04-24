# 文件: backend_api.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import shutil
import os

from Cre_milvus.ColBuilder.visualization import get_cluster_visualization_data
import numpy as np
from Cre_milvus.System.start import load_config, Cre_VectorDataBaseStart_from_config, Cre_Search

# uvicorn backend_api:app --reload 运行
app = FastAPI()

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    upload_dir = "data/upload"
    os.makedirs(upload_dir, exist_ok=True)
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    # 这里可以自动触发向量库构建
    config = load_config()
    Cre_VectorDataBaseStart_from_config(config)
    return {"message": "文件上传并处理完成"}

@app.post("/search")
async def search_api(question: str = Form(...)):
    config = load_config()
    result = Cre_Search(config, question)
    return JSONResponse(content=result)


@app.post("/visualization")
async def cluster_visualization(question: str = Form(...), col_choice: str = Form(...)):
    if col_choice.lower() != "hdbscan":
        return JSONResponse(content={"message": "仅支持HDBSCAN聚类可视化", "data": []})
    config = load_config()
    result = Cre_Search(config, question)
    # 假设 result 里有 clusters，每个 cluster 里有 id, embedding, distance, content
    embeddings = []
    labels = []
    texts = []
    for cluster_id, items in enumerate(result["clusters"].values()):
        for item in items:
            embeddings.append(item["embedding"])
            labels.append(cluster_id)
            texts.append(item.get("content", ""))
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    df = get_cluster_visualization_data(embeddings, labels, texts)
    return df.to_dict(orient="records")

