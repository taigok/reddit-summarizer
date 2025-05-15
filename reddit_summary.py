import os
import praw
from google import genai
from env_loader import load_env
from logging_config import setup_logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

load_env()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "reddit-ultralight-summary-script")


def fetch_reddit_posts(subreddit_name, limit=5, comment_limit=10):
    """
    指定したサブレディットから投稿とコメントを取得します。

    Args:
        subreddit_name (str): サブレディット名。
        limit (int): 取得する投稿数。
        comment_limit (int): 各投稿ごとに取得するコメント数。

    Returns:
        list: 投稿データのリスト。
    """
    logger.info(
        "Fetching posts from subreddit: %s (limit=%d, comment_limit=%d)",
        subreddit_name,
        limit,
        comment_limit,
    )
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    subreddit = reddit.subreddit(subreddit_name)
    post_list = []
    for submission in subreddit.hot(limit=limit):
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments[:comment_limit]]
        post_list.append(
            {
                "id": submission.id,
                "title": submission.title,
                "selftext": submission.selftext,
                "comments": comments,
                "url": f"https://www.reddit.com{submission.permalink}",
            }
        )
    logger.info("Fetched %d posts from subreddit: %s", len(post_list), subreddit_name)
    return post_list


def summarize_post_with_llm(client, post):
    """
    LLM（大規模言語モデル）APIを用いて投稿・コメント全体の要約を生成します。

    Args:
        client: LLM APIクライアント。
        post (dict): 投稿データ。

    Returns:
        str: 要約文。
    """
    logger.info("Summarizing post: %s", post["title"])
    prompt = f"""
RedditのUltralightサブレディットから取得した投稿とコメントのデータです。
この投稿とコメントについて、投稿・コメント全体の要約を300字程度で作成してください。

タイトル: {post['title']}
本文: {post['selftext']}
コメント: {post['comments']}
---
出力形式:
（要約文のみを出力してください。プレフィックスや装飾は不要です）
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=[prompt]
    )
    summary = response.text.strip()
    logger.info("Summary generated for: %s", post["title"])
    return summary


from pydantic import BaseModel
from typing import Optional
from enum import Enum


class ProductType(Enum):
    # Tent & Tarp
    TENT = "Tent"
    TARP = "Tarp"
    TENT_ACCESSORY = "Tent Accessory"

    # Backpack
    BACKPACK = "Backpack"
    SACK_POUCH = "Sack / Waist Pouch"
    BACKPACK_ACCESSORY = "Backpack Accessory"
    TRAVEL_BAG = "Travel Bag / Tote"

    # Sleeping Items
    SLEEPING_BAG = "Sleeping Bag"
    BIVY = "Bivy"
    HAMMOCK = "Hammock"
    MAT = "Sleeping Mat"
    PILLOW = "Pillow"
    GROUNDSHEET = "Groundsheet"
    SLEEP_ACCESSORY = "Sleeping Accessory"

    # Clothing
    TOPS = "Tops"
    TSHIRT = "T-shirt / Shirt"
    SHELL = "Shell"
    INSULATION = "Insulation"
    BOTTOMS = "Bottoms"
    PANTS = "Pants / Shorts"

    # Wear Accessories
    HEADGEAR = "Headgear"
    EYEWEAR = "Eyewear"
    NECKWEAR = "Neckwear"
    GLOVES = "Gloves"
    SOCKS = "Socks"
    SHOES = "Shoes"

    # Bikepacking
    BIKE_BAG = "Bike Bag"
    BIKE_ACCESSORY = "Bike Accessory"

    # Cooker & Accessories
    COOKER = "Cooker"
    CUTLERY = "Cutlery"
    TABLE = "Table"
    STOVE = "Stove / Fuel"
    FIRE = "Firepit"
    BOTTLE_PURIFIER = "Bottle / Water Purifier"

    # Field Gear
    STUFF_SACK = "Stuff Sack"
    FIELD_ACCESSORY = "Field Accessory"
    KNIFE_TOOL = "Knife"
    WALLET = "Wallet"
    UMBRELLA = "Umbrella"
    CRAMPONS = "Crampons"
    EMERGENCY = "Emergency"
    LANTERN_HEADLIGHT = "Lantern / Headlight"

    # Food & Alcohol
    FOOD = "Food"
    ALCOHOL = "Alcohol"

    # Other
    OTHER = "Other"


class Tool(BaseModel):
    """
    Model representing tool/gear information.

    Attributes:
        brand (Optional[str]): Brand name.
        name (str): Tool/product name.
        type (Optional[ProductType]): Product category/type.
    """

    brand: Optional[str]
    name: str
    type: Optional[ProductType] = None


class ToolList(BaseModel):
    """
    道具・ギアのリストを表すモデル。

    Attributes:
        tools (list[Tool]): 道具のリスト。
    """

    tools: list[Tool]


def extract_tools_with_llm(client, post):
    """
    LLM（大規模言語モデル）APIを用いて投稿・コメントから道具・ギアを抽出します。

    Args:
        client: LLM APIクライアント。
        post (dict): 投稿データ。

    Returns:
        list: 道具・ギアのリスト。
    """
    logger.info("Extracting tools from post: %s", post["title"])
    prompt = f"""
RedditのUltralightサブレディットから取得した投稿とコメントのデータです。
この投稿とコメントに登場する「道具・ギア」をリストアップしてください。
それぞれ「ブランド名」「道具名（製品名）」「type（カテゴリ）」の3つの情報を抽出し、以下のJSON形式で返してください。
- ブランド名が不明な場合はnull、または空文字列でも構いません。
- typeは英語でカテゴリ名（例: "Tent", "Cooker" など）。わからない場合はnullや空文字列でも構いません。

出力例:
[
  {{"brand": "Montbell", "name": "U.L.ドームシェルター", "type": "Tent"}},
  {{"brand": null, "name": "チタンマグカップ", "type": "Cooker"}}
]

タイトル: {post['title']}
本文: {post['selftext']}
コメント: {post['comments']}
---
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt],
        config={
            "response_mime_type": "application/json",
            "response_schema": ToolList,
        },
    )
    # response.textはJSON文字列、response.parsedはpydanticモデル
    try:
        tools = response.parsed.tools
        logger.info("Extracted %d tools from: %s", len(tools), post["title"])
    except Exception:
        tools = []
        logger.warning("Failed to extract tools from: %s", post["title"])
    return tools


def summarize_posts_with_llm(posts, max_retries=3, base_wait=60):
    """
    投稿リストを要約し、道具リストも抽出します（LLMモデルを利用）。
    エラー時は指数バックオフでリトライし、失敗した投稿はスキップします。

    Args:
        posts (list): 投稿データのリスト。
        max_retries (int): リトライ最大回数。
        base_wait (int): リトライ間隔の基準（秒）。

    Returns:
        list: 各投稿のタイトル、要約、道具リストを含むリスト。
    """
    logger.info("Summarizing %d posts with LLM...", len(posts))
    client = genai.Client()
    results = []
    for post in posts:
        for attempt in range(1, max_retries + 1):
            try:
                summary = summarize_post_with_llm(client, post)
                tools = extract_tools_with_llm(client, post)
                result = {
                    "id": post["id"],
                    "title": post["title"],
                    "summary": summary,
                    "tools": tools,
                    "url": post.get("url"),
                }
                save_summaries_and_tools_to_db(result)
                results.append(result)
                break  # 成功したらループを抜ける
            except Exception as e:
                logger.warning(
                    f"Error on post '{post['title']}' (attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    wait_time = base_wait * (2 ** (attempt - 1))  # 60, 120, 240...
                    logger.info(f"Retrying after {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # 最終的に失敗した場合はエラー内容を記録してスキップ
                    results.append(
                        {
                            "id": post["id"],
                            "title": post["title"],
                            "summary": f"[ERROR] {e}",
                            "tools": [],
                            "url": post.get("url"),
                        }
                    )
    logger.info("All posts summarized (with retry/skip & backoff).")
    return results


import sqlite3
import os
import time


def init_db(db_path="data/summary.db"):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS summaries (
            id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            url TEXT
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS tools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary_id TEXT,
            brand TEXT,
            name TEXT,
            type TEXT,
            FOREIGN KEY(summary_id) REFERENCES summaries(id),
            UNIQUE(summary_id, name, brand, type)
        )
    """
    )
    conn.commit()
    conn.close()


def save_summaries_and_tools_to_db(summaries, db_path="data/summary.db"):
    # 単一投稿(dict)にも対応
    if isinstance(summaries, dict):
        summaries = [summaries]
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    for item in summaries:
        c.execute(
            "INSERT OR IGNORE INTO summaries (id, title, summary, url) VALUES (?, ?, ?, ?)",
            (item["id"], item["title"], item["summary"], item.get("url")),
        )
        for tool in item["tools"]:
            if not isinstance(tool, dict):
                if hasattr(tool, "model_dump"):
                    tool = tool.model_dump()
                else:
                    tool = dict(tool)
            # ProductType型は文字列に変換
            if isinstance(tool.get("type"), ProductType):
                tool["type"] = tool["type"].value
            c.execute(
                "SELECT COUNT(*) FROM tools WHERE summary_id=? AND name=? AND brand=? AND type=?",
                (item["id"], tool.get("name"), tool.get("brand"), tool.get("type")),
            )
            if c.fetchone()[0] == 0:
                c.execute(
                    "INSERT INTO tools (summary_id, brand, name, type) VALUES (?, ?, ?, ?)",
                    (item["id"], tool.get("brand"), tool.get("name"), tool.get("type")),
                )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    init_db()

    posts = fetch_reddit_posts("Ultralight", limit=2, comment_limit=5)
    summaries = summarize_posts_with_llm(posts)
    for item in summaries:
        tools_list = []
        for tool in item["tools"]:
            d = tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
            if isinstance(d.get("type"), ProductType):
                d["type"] = d["type"].value
            tools_list.append(d)
        item["tools"] = tools_list

    save_summaries_and_tools_to_db(summaries)
    logger.info("DB保存が完了しました")
