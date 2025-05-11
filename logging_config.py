import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Console handler (INFO以上)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s %(message)s'
    )
    console_handler.setFormatter(console_format)

    # File handler (DEBUG以上, ローテーションあり)
    file_handler = RotatingFileHandler(
        'app.log', maxBytes=1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s %(message)s'
    )
    file_handler.setFormatter(file_format)

    # ハンドラを重複追加しない
    if not logger.hasHandlers():
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    else:
        # 既存ハンドラをクリアして再追加
        logger.handlers.clear()
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
