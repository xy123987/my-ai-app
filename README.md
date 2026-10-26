# my-ai-app

基于 Flask + LangChain + ChromaDB + 智谱 API 的 RAG 问答系统。

## 功能
- 文档加载与递归切分（LangChain `RecursiveCharacterTextSplitter`）
- 向量化存储（ChromaDB 持久化）
- 检索增强生成（RAG）对话
- 对话历史持久化（MySQL）

## 运行方式
1. 复制 `.env.example` 为 `.env`，填入 `ZHIPU_API_KEY` 和 `MYSQL_PASSWORD`
2. 安装依赖：`pip install -r requirements.txt`
3. 初始化向量库：`python rag_vector.py`
4. 单次查询测试：`python rag_query.py`
5. 启动 Web 服务：`python app.py`

## 接口说明
- `POST /chat`：提交问题，返回 AI 回答
  ```json
  {"message": "你的问题"}