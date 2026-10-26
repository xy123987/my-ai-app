import os
import logging
from pathlib import Path

from dotenv import load_dotenv
import chromadb
import zhipuai

# 1. 绝对路径加载规范
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("环境变量 ZHIPU_API_KEY 未定义，请检查 .env 文件。")

# 2. 获取 logger（不在此处配置，避免污染导入方）
logger = logging.getLogger("RAG_Query")

# 3. 初始化客户端
client = zhipuai.ZhipuAI(api_key=API_KEY)
chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
collection = chroma_client.get_or_create_collection(name="my_rag_collection")

def query_rag(user_query: str, n_results: int = 3) -> str:
    try:
        # 4. 将用户问题向量化
        logger.info("正在向量化用户问题...")
        response = client.embeddings.create(
            model="embedding-2",
            input=user_query,
        )
        query_vector = response.data[0].embedding

        # 5. 在向量库中检索最相似的片段
        logger.info(f"正在检索最相似的 {n_results} 个片段...")
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=n_results
        )

        # 6. 提取检索结果
        retrieved_chunks = results['documents'][0]
        context = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])
        logger.info(f"检索到 {len(retrieved_chunks)} 个相关片段。")

        # 7. 拼接 Prompt
        prompt = f"""请严格根据以下参考资料回答用户问题。如果资料中不包含答案，请回答“根据已知资料无法回答”。

【参考资料】
{context}

【用户问题】
{user_query}"""

        # 8. 调用大模型生成回答
        logger.info("正在调用大模型生成回答...")
        llm_response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        answer = llm_response.choices[0].message.content
        logger.info("✅ 回答生成完毕。")
        return answer

    except Exception as e:
        logger.error(f"检索或生成过程发生异常: {e}", exc_info=True)
        return "处理请求时发生内部错误。"

if __name__ == "__main__":
    # 仅在此处配置日志
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
    test_query = "请概括一下这篇文档的核心内容？"
    final_answer = query_rag(test_query, n_results=3)
    print("\n" + "="*20 + " 最终回答 " + "="*20)
    print(final_answer)
    print("="*50)