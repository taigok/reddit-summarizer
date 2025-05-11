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


def summarize_posts_with_llm(posts):
    """
    投稿リストを要約し、道具リストも抽出します（LLMモデルを利用）。

    Args:
        posts (list): 投稿データのリスト。

    Returns:
        list: 各投稿のタイトル、要約、道具リストを含むリスト。
    """
    logger.info("Summarizing %d posts with LLM...", len(posts))
    client = genai.Client()
    results = []
    for post in posts:
        summary = summarize_post_with_llm(client, post)
        tools = extract_tools_with_llm(client, post)
        results.append({"title": post["title"], "summary": summary, "tools": tools})
    logger.info("All posts summarized.")
    return results


if __name__ == "__main__":
    import json

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
me == "master":
            return "main"
        else:
            return "other"

    branch_name = get_current_branch()
    branch_type = get_branch_type(branch_name)

    posts = fetch_reddit_posts("Ultralight", limit=10, comment_limit=10)
    summaries = summarize_posts_with_llm(posts)
    # Toolはpydanticモデルなのでdict化
    for item in summaries:
        item["tools"] = [
            tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
            for tool in item["tools"]
        ]
        item["branch_name"] = branch_name
        item["branch_type"] = branch_type
    logger.info(
        "Summary output: %s", json.dumps(summaries, ensure_ascii=False, indent=2)
    )
