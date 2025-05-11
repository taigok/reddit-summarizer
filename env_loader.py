from pathlib import Path
from dotenv import load_dotenv
import os

def load_env():
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        print("[Warning] .env file not found. Using system environment variables only.")

# 利用例:
# from env_loader import load_env
# load_env()
# その後 os.getenv(...) で値が取得できます
