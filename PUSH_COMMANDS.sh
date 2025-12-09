#!/bin/bash
# GitHubへのプッシュスクリプト

cd "/Users/ryosukenakamizo/Documents/Cursor/Azure RAG"

echo "=== 1. リモートリポジトリの設定 ==="
git remote remove origin 2>/dev/null
git remote add origin https://github.com/RyossukeNakamizo/azure-rag-embedding.git
git remote -v

echo -e "\n=== 2. すべてのファイルを追加 ==="
git add embedding.py config.py requirements.txt README.md .gitignore .env.example DEBUG_REPORT.md GITHUB_SETUP.md PUSH_INSTRUCTIONS.md

echo -e "\n=== 3. コミット状態の確認 ==="
git status --short

echo -e "\n=== 4. コミット ==="
git commit -m "feat: Azure AI Search向けRAG検索の基盤となる非同期Embeddingクライアント

- 非同期Embeddingクライアント (embedding.py)
- 環境変数管理モジュール (config.py)
- ドキュメントとセットアップファイル"

echo -e "\n=== 5. GitHubへプッシュ ==="
git push -u origin main

echo -e "\n=== 完了 ==="

