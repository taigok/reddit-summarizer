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
            }
        )
    return posts


# --- Geminiで要約・道具リスト生成 ---
def summarize_with_gemini(posts):
    client = genai.Client()
    results = []
    for post in posts:
        prompt = f"""
RedditのUltralightサブレディットから取得した投稿とコメントのデータです。
この投稿とコメントについて、
1. 投稿・コメント全体の要約を300字程度で作成してください。
2. 投稿やコメントに登場する道具やギアのリストを抽出してください。

タイトル: {post['title']}
本文: {post['selftext']}
コメント: {post['comments']}
---
出力形式:
要約: ...\n道具リスト: ...
"""
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=[prompt]
        )
        results.append({"title": post["title"], "summary": response.text.strip()})
    return results


if __name__ == "__main__":
    posts = fetch_reddit_posts("Ultralight", limit=3, comment_limit=5)
    summaries = summarize_with_gemini(posts)
    for item in summaries:
        print(f"\n---\nタイトル: {item['title']}\n{item['summary']}\n")
