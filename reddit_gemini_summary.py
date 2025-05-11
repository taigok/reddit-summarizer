import os
import praw
from google import genai
from env_loader import load_env

load_env()

REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "reddit-ultralight-summary-script")


# --- Redditから投稿とコメント取得 ---
def fetch_reddit_posts(subreddit_name, limit=5, comment_limit=10):
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


# --- Geminiで要約・道具リスト生成（要約・道具リストを別リクエスト） ---
def summarize_post_with_gemini(client, post):
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
    brand: Optional[str]
    name: str


class ToolList(BaseModel):
    tools: list[Tool]


def extract_tools_with_gemini(client, post):
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


def summarize_with_gemini(posts):
    client = genai.Client()
    results = []
    for post in posts:
        summary = summarize_post_with_gemini(client, post)
        tools = extract_tools_with_gemini(client, post)
        results.append({"title": post["title"], "summary": summary, "tools": tools})
    return results


if __name__ == "__main__":
    import json

    posts = fetch_reddit_posts("Ultralight", limit=3, comment_limit=5)
    summaries = summarize_with_gemini(posts)
    # Toolはpydanticモデルなのでdict化
    for item in summaries:
        item["tools"] = [
            tool.dict() if hasattr(tool, "dict") else dict(tool)
            for tool in item["tools"]
        ]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
