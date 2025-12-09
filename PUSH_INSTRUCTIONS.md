# GitHubへのプッシュ手順

## 方法1: ブラウザで作成（最も簡単）

### ステップ1: GitHubでリポジトリを作成

1. https://github.com/new を開く
2. 以下の情報を入力：
   - **Repository name**: `azure-rag-embedding`
   - **Description**: `Azure AI Search向けRAG検索の基盤となる非同期Embeddingクライアント`
   - **Visibility**: Public または Private を選択
   - ⚠️ **「Add a README file」のチェックを外す**（既に存在するため）
3. 「Create repository」をクリック

### ステップ2: リモートリポジトリを追加してプッシュ

GitHubで作成したリポジトリのページに表示されるコマンドをコピーして実行：

```bash
cd "/Users/ryosukenakamizo/Documents/Cursor/Azure RAG"

# リモートリポジトリを追加
git remote add origin https://github.com/YOUR_USERNAME/azure-rag-embedding.git

# プッシュ
git push -u origin main
```

または、以下のコマンドを実行してください（リポジトリURLを置き換えてください）：

```bash
cd "/Users/ryosukenakamizo/Documents/Cursor/Azure RAG"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

## 方法2: GitHub CLIを使用（自動化）

### 前提条件
GitHub CLI（gh）がインストールされ、認証済みである必要があります。

### インストール（未インストールの場合）

```bash
# Homebrewでインストール
brew install gh

# 認証
gh auth login
```

### リポジトリ作成とプッシュ

```bash
cd "/Users/ryosukenakamizo/Documents/Cursor/Azure RAG"

# GitHubリポジトリを作成（public）
gh repo create azure-rag-embedding \
  --public \
  --source=. \
  --remote=origin \
  --description="Azure AI Search向けRAG検索の基盤となる非同期Embeddingクライアント"

# プッシュ
git push -u origin main
```

または、privateリポジトリとして作成する場合：

```bash
gh repo create azure-rag-embedding \
  --private \
  --source=. \
  --remote=origin \
  --description="Azure AI Search向けRAG検索の基盤となる非同期Embeddingクライアント"

git push -u origin main
```

## 認証エラーが発生した場合

### Personal Access Token (PAT) を使用

1. GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 「Generate new token」をクリック
3. `repo` スコープを選択
4. トークンをコピー
5. プッシュ時にパスワードの代わりに使用

### SSH鍵を使用

```bash
# SSH鍵を生成（未作成の場合）
ssh-keygen -t ed25519 -C "your_email@example.com"

# SSH鍵をGitHubに追加
cat ~/.ssh/id_ed25519.pub
# → この内容をGitHub Settings → SSH and GPG keys に追加

# SSH URLを使用
git remote add origin git@github.com:YOUR_USERNAME/azure-rag-embedding.git
git push -u origin main
```

## トラブルシューティング

### エラー: `remote origin already exists`
```bash
git remote remove origin
git remote add origin YOUR_REPO_URL
```

### エラー: `failed to push some refs`
```bash
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## 現在のコミット状態

✅ 2つのコミットが準備完了:
1. feat: Azure AI Search向けRAG検索の基盤となる非同期Embeddingクライアントと環境変数管理モジュール
2. docs: デバッグレポートとGitHubセットアップガイドを追加

## 含まれるファイル

- embedding.py
- config.py
- requirements.txt
- README.md
- .gitignore
- .env.example
- DEBUG_REPORT.md
- GITHUB_SETUP.md

すべての準備が完了しています。上記の手順に従ってプッシュしてください！

