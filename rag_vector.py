import os
import logging
from pathlib import Path

from dotenv import load_dotenv
import chromadb
import zhipuai

from rag_demo import load_document, split_text

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("环境变量 ZHIPU_API_KEY 未定义")

logger = logging.getLogger("RAG_Vector")
client = zhipuai.ZhipuAI(api_key=API_KEY)
chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
collection = chroma_client.get_or_create_collection(name="my_rag_collection")

BATCH_SIZE = 20


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

    text = load_document(BASE_DIR / "test.txt")
    chunks = split_text(text)
    logger.info(f"✅ 切分为 {len(chunks)} 个片段")

    # 批量调用 Embedding API，降低网络请求次数
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        try:
            response = client.embeddings.create(model="embedding-2", input=batch_chunks)
            for j, data in enumerate(response.data):
                idx = i + j
                collection.add(
                    ids=[f"id_{idx}"],
                    documents=[batch_chunks[j]],
                    embeddings=[data.embedding],
                )
            logger.info(f"已处理 {i + len(batch_chunks)}/{len(chunks)}")
        except Exception as e:
            logger.error(f"❌ 批次 {i} 向量化失败: {e}", exc_info=True)
            continue

    logger.info("✅ 全部向量化完成并入库。")