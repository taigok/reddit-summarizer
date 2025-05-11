import os
import praw
from google import genai
from env_loader import load_env

# .envから環境変数をロード
load_env()

# --- 環境変数からAPIキー取得 ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "reddit-ultralight-summary-script")

# Google Gemini APIキーは環境変数 GOOGLE_API_KEY にセットしてください
# 例: export GOOGLE_API_KEY='your-gemini-api-key'

# --- Redditから投稿とコメント取得 ---
def fetch_reddit_posts(subreddit_name, limit=5, comment_limit=10):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.hot(limit=limit):
        submission.comments.replace_more(limit=0)
        comments = [comment.body for comment in submission.comments[:comment_limit]]
        posts.append({
            'title': submission.title,
            'selftext': submission.selftext,
            'comments': comments
        })
    return posts

# --- Geminiで要約・道具リスト生成 ---
def summarize_with_gemini(posts):
    client = genai.Client()
    prompt = """
RedditのUltralightサブレディットから取得した投稿とコメントのデータです。
1. 全体の要約を300字程度で作成してください。
2. 投稿やコメントに登場する道具やギアのリストを抽出してください。

データ:
"""
    for post in posts:
        prompt += f"\nタイトル: {post['title']}\n本文: {post['selftext']}\nコメント: {post['comments']}\n"
    prompt += "\n---\n出力形式:\n要約: ...\n道具リスト: ..."
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text

if __name__ == "__main__":
    posts = fetch_reddit_posts("Ultralight", limit=3, comment_limit=5)
    summary = summarize_with_gemini(posts)
    print(summary)
