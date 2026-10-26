import logging
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 仅获取 logger，不在模块级别调用 basicConfig，避免污染主程序日志配置
logger = logging.getLogger("RAG_Demo")

BASE_DIR = Path(__file__).resolve().parent


def load_document(file_path: str | Path) -> str:
    """
    读取本地文本文件

    :param file_path: 文件路径（支持字符串或 Path 对象）
    :return: 文件全部文本内容
    :raises FileNotFoundError: 文件不存在时抛出
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def split_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    """
    使用 LangChain 递归字符切分器

    :param text: 待切分的长文本
    :param chunk_size: 每个片段的最大字符数
    :param overlap: 相邻片段的重叠字符数
    :return: 片段列表
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    # 主程序运行时，自行配置日志
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')

    # 使用绝对路径，确保无论从何处运行都能找到文件
    text = load_document(BASE_DIR / "test.txt")
    logger.info(f"✅ 成功读取文档，总字数: {len(text)}")

    chunks = split_text(text)
    logger.info(f"✅ 成功切分为 {len(chunks)} 个片段")

    for i, chunk in enumerate(chunks[:2]):
        # 截断预览，避免日志过长
        preview = chunk[:100].replace("\n", " ")
        logger.info(f"--- 第 {i + 1} 块预览 ---\n{preview}...")