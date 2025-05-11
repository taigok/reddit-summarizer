import os
import praw
from google import genai
from env_loader import load_env

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
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.hot(limit=limit):
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments[:comment_limit]]
        posts.append(
            {
                "title": submission.title,
                "selftext": submission.selftext,
                "comments": comments,
                "url": f"https://www.reddit.com{submission.permalink}",
            }
        )
    return posts


def summarize_post_with_llm(client, post):
    """
    LLM（大規模言語モデル）APIを用いて投稿・コメント全体の要約を生成します。

    Args:
        client: LLM APIクライアント。
        post (dict): 投稿データ。

    Returns:
        str: 要約文。
    """
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
    return response.text.strip()


from pydantic import BaseModel
from typing import Optional


class Tool(BaseModel):
    """
    道具・ギアの情報を表すモデル。

    Attributes:
        brand (Optional[str]): ブランド名。
        name (str): 道具名（製品名）。
    """

    brand: Optional[str]
    name: str


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
    prompt = f"""
RedditのUltralightサブレディットから取得した投稿とコメントのデータです。
この投稿とコメントに登場する「道具・ギア」をリストアップしてください。
それぞれ「ブランド名」と「道具名（製品名）」の2つの情報を抽出し、以下のJSON形式で返してください。
ブランド名が不明な場合はnull、または空文字列でも構いません。

出力例:
[
  {{"brand": "Montbell", "name": "U.L.ドームシェルター"}},
  {{"brand": null, "name": "チタンマグカップ"}}
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
    except Exception:
        tools = []
    return tools


def summarize_posts_with_llm(posts):
    """
    投稿リストを要約し、道具リストも抽出します（LLMモデルを利用）。

    Args:
        posts (list): 投稿データのリスト。

    Returns:
        list: 各投稿のタイトル、要約、道具リストを含むリスト。
    """
    client = genai.Client()
    results = []
    for post in posts:
        summary = summarize_post_with_llm(client, post)
        tools = extract_tools_with_llm(client, post)
        results.append({"title": post["title"], "summary": summary, "tools": tools})
    return results


if __name__ == "__main__":
    import json

    posts = fetch_reddit_posts("Ultralight", limit=3, comment_limit=5)
    summaries = summarize_posts_with_llm(posts)
    # Toolはpydanticモデルなのでdict化
    for item in summaries:
        item["tools"] = [
            tool.dict() if hasattr(tool, "dict") else dict(tool)
            for tool in item["tools"]
        ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
