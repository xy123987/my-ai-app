import os
import logging
from pathlib import Path

import jieba
from dotenv import load_dotenv
import chromadb
import zhipuai
from rank_bm25 import BM25Okapi

# 1. 基础配置
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
API_KEY = os.getenv("ZHIPU_API_KEY")
client = zhipuai.ZhipuAI(api_key=API_KEY)

chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
collection = chroma_client.get_or_create_collection(name="my_rag_collection")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("Evaluate")

# 获取全量数据构建 BM25
_stored = collection.get()
DOC_IDS = _stored["ids"]
DOC_TEXTS = _stored["documents"]
DOC_MAP = dict(zip(DOC_IDS, DOC_TEXTS))
BM25_CORPUS = [list(jieba.lcut(text)) for text in DOC_TEXTS]
bm25 = BM25Okapi(BM25_CORPUS)

# 2. 定义测试集（严格基于你的 PDF 内容，关键词100%存在）
TEST_SET = [
    {"query": "这门课的指导老师是谁？", "expected_keyword": "朱坤志"},
    {"query": "11月27日之前需要提交什么？", "expected_keyword": "课程实践报告"},
    {"query": "如果有任意2个模型完全相同，会被判为多少分？", "expected_keyword": "0分"},
    {"query": "报告撰写的分数占比是多少？", "expected_keyword": "50分"},
    {"query": "封面、目录、评审表单面打印，正文怎么打印？", "expected_keyword": "双面打印"},
    {"query": "可以用Word的自动编号功能吗？", "expected_keyword": "不使用"}
]

def get_vector_retrieval(query, top_k=3):
    response = client.embeddings.create(model="embedding-2", input=query)
    query_vector = response.data[0].embedding
    results = collection.query(query_embeddings=[query_vector], n_results=top_k)
    return results["ids"][0]

def get_hybrid_retrieval(query, top_k=3):
    # 向量召回 Top 10
    response = client.embeddings.create(model="embedding-2", input=query)
    vector_results = collection.query(query_embeddings=[response.data[0].embedding], n_results=10)
    vector_ids = vector_results["ids"][0]
    
    # BM25 召回 Top 10
    query_tokens = list(jieba.lcut(query))
    bm25_scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    bm25_ids = [DOC_IDS[i] for i in top_indices]
    
    # RRF 融合
    fused_scores = {}
    for rank, doc_id in enumerate(vector_ids, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
        
    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
    return ranked_ids

def evaluate():
    vector_hits = 0
    hybrid_hits = 0
    total = len(TEST_SET)
    
    logger.info(f"开始评测，共 {total} 条测试数据...")
    
    for i, item in enumerate(TEST_SET, 1):
        query = item["query"]
        keyword = item["expected_keyword"]
        
        # 测试纯向量
        v_ids = get_vector_retrieval(query, top_k=3)
        v_texts = "".join([DOC_MAP.get(i, "") for i in v_ids])
        if keyword in v_texts:
            vector_hits += 1
            logger.info(f"[{i}] 向量检索命中！关键词: {keyword}")
        else:
            logger.info(f"[{i}] 向量检索未命中。查询: {query}")
            
        # 测试混合检索
        h_ids = get_hybrid_retrieval(query, top_k=3)
        h_texts = "".join([DOC_MAP.get(i, "") for i in h_ids])
        if keyword in h_texts:
            hybrid_hits += 1
            logger.info(f"[{i}] 混合检索命中！关键词: {keyword}")
        else:
            logger.info(f"[{i}] 混合检索未命中。查询: {query}")

    logger.info("="*30)
    logger.info(f"向量检索 Top-3 命中率: {vector_hits}/{total} ({(vector_hits/total)*100:.1f}%)")
    logger.info(f"混合检索 Top-3 命中率: {hybrid_hits}/{total} ({(hybrid_hits/total)*100:.1f}%)")
    logger.info("="*30)

if __name__ == "__main__":
    evaluate()