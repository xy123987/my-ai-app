import os
import logging
from pathlib import Path

import pymysql
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

logger = logging.getLogger("DB")


def get_connection() -> pymysql.connections.Connection:
    """
    从环境变量读取数据库配置，返回 MySQL 连接对象

    :return: pymysql 连接对象
    :raises pymysql.MySQLError: 连接失败时抛出
    """
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE", "ai_app"),
        charset="utf8mb4",
    )