import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify

from db import get_connection
# 直接复用你写好的 RAG 核心逻辑，避免代码重复
from rag_query import query_rag 

# ==================== 1. 配置加载与校验 ====================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

API_KEY = os.getenv("ZHIPU_API_KEY")
if not API_KEY:
    raise ValueError("环境变量 ZHIPU_API_KEY 未定义，请检查 .env 文件。")

# ==================== 2. 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("LLMApp")

# ==================== 3. Flask 应用初始化 ====================
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


# ==================== 4. Agent 类定义 ====================
class ChatAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def call_llm(self, prompt: str) -> str:
        """
        调用核心 RAG 引擎：
        1. 向量化用户问题
        2. 从 ChromaDB 检索相关文档
        3. 拼接 Prompt 并调用大模型
        """
        try:
            # 直接调用 rag_query.py 中的 RAG 流程
            return query_rag(user_query=prompt, n_results=3)
        except Exception as e:
            logger.error(f"调用RAG系统时发生异常: {e}", exc_info=True)
            return f"请求失败: {e}"

    def save_to_db(self, user_msg: str, ai_reply: str) -> None:
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (user_msg, ai_reply) VALUES (%s, %s)",
                (user_msg, ai_reply),
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info("✅ 对话已成功存入数据库")
        except Exception as e:
            logger.error(f"❌ 数据库写入失败: {e}", exc_info=True)


# ==================== 5. 实例化 Agent ====================
agent = ChatAgent(API_KEY)


# ==================== 6. 接口路由 ====================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "message" not in data:
        logger.warning("收到无效请求，缺少 message 字段")
        return jsonify({"error": "请提供message字段"}), 400

    user_msg = data["message"]
    logger.info(f"收到用户请求：{user_msg}")

    # 此时调用的是 RAG 引擎
    reply = agent.call_llm(user_msg)
    agent.save_to_db(user_msg, reply)

    return jsonify({"reply": reply})


@app.route("/history", methods=["GET"])
def get_history():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT user_msg, ai_reply, create_time FROM chat_history ORDER BY id DESC LIMIT 10"
        )
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        history_list = []
        for row in results:
            history_list.append({
                "user_msg": row[0],
                "ai_reply": row[1],
                "time": str(row[2]),
            })
        return jsonify({"history": history_list})
    except Exception as e:
        logger.error(f"读取历史记录失败: {e}", exc_info=True)
        return jsonify({"error": "服务器内部错误，请稍后重试"}), 500


# ==================== 7. 程序入口 ====================
if __name__ == "__main__":
    logger.info("AI服务器启动...")
    app.run(debug=True)