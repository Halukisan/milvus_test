import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# streamlit run frontend.py
st.title("智能向量检索系统")

# 上传文件夹
st.header("上传数据文件夹")
uploaded_files = st.file_uploader("选择文件夹中的文件（支持csv, md, pdf, txt）", accept_multiple_files=True)

if st.button("上传并构建向量库"):
    files = [("files", (file.name, file, file.type)) for file in uploaded_files]
    response = requests.post("http://localhost:8000/upload", files=files)
    st.write(response.json()["message"])

# 检索参数
st.header("检索参数设置")
question = st.text_input("请输入检索问题")
col_choice = st.selectbox("聚类算法", ["hdbscan", "kmeans"])

if st.button("开始检索与可视化"):
    # 检索
    response = requests.post(
        "http://localhost:8000/search",
        data={"question": question, "col_choice": col_choice}
    )
    result = response.json()
    st.write(result.get("message", ""))
    # 只有HDBSCAN才可视化
    if col_choice.lower() == "hdbscan":
        vis_response = requests.post(
            "http://localhost:8000/visualization",
            data={"question": question, "col_choice": col_choice}
        )
        vis_data = vis_response.json()
        if isinstance(vis_data, dict) and "data" in vis_data and not vis_data["data"]:
            st.info("仅支持HDBSCAN聚类可视化")
        elif vis_data:
            df = pd.DataFrame(vis_data)
            fig = px.scatter(
                df, x="x", y="y", color="cluster", hover_data=["text"],
                title="HDBSCAN聚类可视化（UMAP降维）"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无可视化数据")
    else:
        st.info("当前聚类算法不支持可视化")