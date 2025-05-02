import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="智能向量检索系统", layout="wide")
st.title("🔍 智能向量检索系统")
st.markdown("---")

# 配置参数设置
with st.expander("⚙️ 配置参数设置", expanded=True):
    with st.form("config_form"):
        st.subheader("Milvus 配置")
        col1, col2, col3= st.columns(3)
        with col1:
            milvus_host = st.text_input("Milvus Host", value="127.0.0.1")
            milvus_user = st.text_input("Milvus User", value="haluki")
            vector_name = st.text_input("Vector DB Name", value="multi_search")
        with col2:
            milvus_port = st.text_input("Milvus Port", value="19530")
            milvus_password = st.text_input("Milvus Password", value="yourpassword")
            collection_name = st.text_input("Collection Name", value="Test_one")
        with col3:
            index_name = st.selectbox("Index Name", ["IVF_FLAT", "HNSW", "HNSW_SQ8"])
            replica_num = st.number_input("Replica Num", value=1, min_value=1)
            url_split = st.selectbox("是否启用URL切分", ["True", "False"])
            index_ddevice = st.selectbox("Index Device", ["cpu", "gpu"])


        st.subheader("ElasticSearch 配置")
        es_host = st.text_input("ElasticSearch Host+Port", value="http://127.0.0.1:9200")
        es_index = st.text_input("ElasticSearch Index", value="my_index")

        st.subheader("redis 配置")
        re_host = st.text_input("Redis Host", value="http://127.0.0.1",key="re_host")
        re_port = st.text_input("Redis Port", value="6379",key="re_port")

        st.subheader("检索参数")
        search_top_k = st.number_input("Search Top K", value=20, min_value=1)
        search_col_choice = st.selectbox("Search Col Choice", ["hdbscan", "kmeans"])
        search_reorder_strategy = st.selectbox("Search Reorder Strategy", ["distance", "cluster_size", "cluster_center"])

        submitted = st.form_submit_button("💾 保存配置")
        if submitted:
            config_data = {
                "milvus": {
                    "host": milvus_host,
                    "port": milvus_port,
                    "user": milvus_user,
                    "password": milvus_password,
                    "vector_name": vector_name,
                    "collection_name": collection_name,
                    "index_name": index_name,
                    "replica_num": replica_num,
                    "index_device": index_ddevice
                },
                "system": {
                    "url_split": url_split == "True"
                },
                "elasticsearch": {
                    "host": es_host,
                    "index": es_index
                },
                "redis": {
                    "host": re_host,
                    "port": re_port
                },
                "search": {
                    "top_k": search_top_k,
                    "col_choice": search_col_choice,
                    "reorder_strategy": search_reorder_strategy
                }
            }
            response = requests.post("http://localhost:8000/update_config", json=config_data)
            try:
                if response.status_code == 200:
                    msg = response.json().get("message", "配置已保存")
                    st.success(msg)
                else:
                    st.error(f"配置保存失败，状态码: {response.status_code}")
            except Exception as e:
                st.error(f"配置保存失败，后端未返回有效JSON: {e}")

st.markdown("---")

# 上传文件夹
with st.expander("📁 上传数据文件夹", expanded=True):
    st.info("请全选文件夹下所有文件上传，并输入一个文件夹名，系统会自动保存到该目录。")
    folder_name = st.text_input("请输入目标文件夹名（如20240501）")
    uploaded_files = st.file_uploader(
        "选择文件夹中的文件（支持csv, md, pdf, txt）", accept_multiple_files=True, type=["csv", "md", "pdf", "txt","JPEG"]
    )
    if st.button("⬆️ 上传并构建向量库"):
        if not folder_name:
            st.warning("请先输入目标文件夹名。")
        elif not uploaded_files:
            st.warning("请先选择要上传的文件。")
        else:
            # 1. 上传文件
            files = [("files", (file.name, file, file.type)) for file in uploaded_files]
            response = requests.post(
                "http://localhost:8000/upload",
                params={"folder_name": folder_name},
                files=files
            )
            # 2. 更新配置文件中的 data_location 字段
            config_update = {
                "data": {
                    "data_location": f"../data/upload/{folder_name}"
                }
            }
            requests.post("http://localhost:8000/update_config", json=config_update)
            try:
                st.write(response.json().get("message", "上传完成"))
            except Exception:
                st.error("后端未返回有效JSON，请检查后端日志。")

st.markdown("---")

# 检索参数
with st.expander("🔎 检索与可视化", expanded=True):
    question = st.text_input("请输入检索问题")
    col_choice = st.selectbox("聚类算法", ["hdbscan", "kmeans"])
    if st.button("🚀 开始检索与可视化"):
        if not question:
            st.warning("请输入检索问题！")
        else:
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