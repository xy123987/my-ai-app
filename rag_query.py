import os
import logging
import time
from pathlib import Path

import jieba
from dotenv import load_dotenv
import chromadb
import zhipuai
from rank_bm25 import BM25Okapi

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

# 4. 构建 BM25 索引：取出库中全部文档并分词
RECALL_TOP_K = 10
RRF_K = 60

_stored = collection.get()
DOC_IDS: list[str] = _stored["ids"]
DOC_TEXTS: list[str] = _stored["documents"]
DOC_MAP = dict(zip(DOC_IDS, DOC_TEXTS))

BM25_CORPUS = [list(jieba.lcut(text)) for text in DOC_TEXTS]
bm25 = BM25Okapi(BM25_CORPUS) if BM25_CORPUS else None


def query_rag(user_query: str, n_results: int = 3) -> str:
    try:
        # 5. 向量检索：取 Top 10
        vector_start = time.perf_counter()
        response = client.embeddings.create(
            model="embedding-2",
            input=user_query,
        )
        query_vector = response.data[0].embedding

        vector_results = collection.query(
            query_embeddings=[query_vector],
            n_results=RECALL_TOP_K
        )
        vector_ids = vector_results["ids"][0]
        vector_time = time.perf_counter() - vector_start
        logger.info(f"向量检索耗时 {vector_time:.4f}s，命中 {len(vector_ids)} 个片段。")

        # 6. BM25 检索：取 Top 10
        bm25_start = time.perf_counter()
        if bm25 is not None:
            query_tokens = list(jieba.lcut(user_query))
            bm25_scores = bm25.get_scores(query_tokens)
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:RECALL_TOP_K]
            bm25_ids = [DOC_IDS[i] for i in top_indices]
        else:
            bm25_ids = []
            logger.warning("BM25 索引为空，跳过关键词召回。")
        bm25_time = time.perf_counter() - bm25_start
        logger.info(f"BM25 检索耗时 {bm25_time:.4f}s，命中 {len(bm25_ids)} 个片段。")

        # 7. RRF 融合两路召回结果
        rrf_start = time.perf_counter()
        fused_scores: dict[str, float] = {}
        for rank, doc_id in enumerate(vector_ids, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, doc_id in enumerate(bm25_ids, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:n_results]
        rrf_time = time.perf_counter() - rrf_start
        logger.info(
            f"RRF 融合耗时 {rrf_time:.4f}s，融合 {len(fused_scores)} 个候选，"
            f"选取前 {len(ranked_ids)} 个片段。"
        )

        # 8. 拼接上下文
        context = "\n".join(f"- {DOC_MAP.get(doc_id, '')}" for doc_id in ranked_ids)

        # 9. 拼接 Prompt
        prompt = f"""请严格根据以下参考资料回答用户问题。如果资料中不包含答案，请回答“根据已知资料无法回答”。

【参考资料】
{context}

【用户问题】
{user_query}"""

        # 10. 调用大模型生成回答
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
