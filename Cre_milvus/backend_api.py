# 文件: backend_api.py
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import shutil
import os
import yaml
from ColBuilder.visualization import get_cluster_visualization_data
import numpy as np
from System.start import load_config, Cre_VectorDataBaseStart_from_config, Cre_Search

# uvicorn backend_api:app --reload 运行
app = FastAPI()

@app.post("/update_config")
async def update_config(request: Request):
    data = await request.json()
    with open("../Cre_milvus/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.update(data)
    with open("../Cre_milvus/config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
    return {"message": "配置已更新"}

@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), folder_name: str = None):
    if not folder_name:
        return {"message": "未指定目标文件夹名"}
    upload_dir = f"data/upload/{folder_name}"
    os.makedirs(upload_dir, exist_ok=True)
    for file in files:
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    # 读取配置并处理数据
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

