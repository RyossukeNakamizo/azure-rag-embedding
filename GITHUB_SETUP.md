# GitHubへのプッシュ手順

## 現在の状態

✅ ローカルgitリポジトリの初期化完了  
✅ 初回コミット完了  
✅ mainブランチへの変更完了

## GitHubリポジトリへのプッシュ手順

### 1. GitHubで新しいリポジトリを作成

1. https://github.com/new にアクセス
2. リポジトリ名を入力（例: `azure-rag-embedding`）
3. 説明を入力（例: `Azure AI Search向けRAG検索の基盤となる非同期Embeddingクライアント`）
4. Public または Private を選択
5. **「Initialize this repository with a README」のチェックを外す**（既にREADME.mdがあるため）
6. 「Create repository」をクリック

### 2. リモートリポジトリを追加

GitHubで作成したリポジトリのURLを使用して、以下のコマンドを実行：

```bash
cd "/Users/ryosukenakamizo/Documents/Cursor/Azure RAG"

# HTTPSの場合
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# または、SSHの場合
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

### 3. GitHubにプッシュ

```bash
git push -u origin main
```

### 4. 認証

初回プッシュ時に認証が求められます：

#### HTTPSの場合
- Personal Access Token (PAT) を使用
- GitHub Settings → Developer settings → Personal access tokens → Generate new token
- `repo` スコープを選択

#### SSHの場合
- SSH鍵が設定済みであることを確認
- 未設定の場合: https://docs.github.com/ja/authentication/connecting-to-github-with-ssh

## コミット内容

### 追加されたファイル

- `embedding.py` - 非同期Embeddingクライアント
- `config.py` - 環境変数管理モジュール
- `requirements.txt` - 依存関係
- `README.md` - ドキュメント
- `.gitignore` - Git除外設定
- `.env.example` - 環境変数テンプレート

### コミットメッセージ

```
feat: Azure AI Search向けRAG検索の基盤となる非同期Embeddingクライアントと環境変数管理モジュール

- 非同期Embeddingクライアント (embedding.py)
  - AsyncAzureOpenAI使用による高性能なEmbedding生成
  - バッチ処理対応（最大16テキスト/リクエスト）
  - トークン数ベースの自動バッチ分割
  - 能動的なレート制限管理
  - Exponential backoffリトライロジック
  - 詳細なメトリクス収集（レイテンシ、トークン数）
  - 長文チャンク分割機能（複数の集約方法対応）
  - retry-afterヘッダー対応
  - APITimeoutError処理

- 環境変数管理モジュール (config.py)
  - Pydantic Settingsによる型安全な設定管理
  - 詳細なバリデーション（エンドポイントURL、APIキー長、数値範囲）
  - シングルトンパターンによる設定の再利用
  - Azure OpenAI設定
  - Azure AI Search設定（将来の拡張用）
  - レート制限設定
  - 設定検証ユーティリティ

- ドキュメント
  - README.md: セットアップ手順と使用例
  - .env.example: 環境変数テンプレート
  - requirements.txt: 依存関係管理
```

## 推奨されるGitHubリポジトリ設定

### リポジトリ名の例
- `azure-rag-embedding`
- `azure-openai-embedding-client`
- `rag-embedding-foundation`

### トピック（Topics）の例
- `azure-openai`
- `embedding`
- `rag`
- `azure-ai-search`
- `python`
- `async`
- `pydantic`

### ライセンス
- MIT License（推奨）
- Apache License 2.0
- または、プロジェクトの要件に応じて選択

## 次のステップ

1. GitHubリポジトリを作成
2. リモートリポジトリを追加
3. プッシュ
4. （オプション）GitHub Actionsでテスト自動化
5. （オプション）PyPIへの公開

## トラブルシューティング

### エラー: `remote origin already exists`
```bash
git remote remove origin
git remote add origin YOUR_REPO_URL
```

### エラー: `failed to push some refs`
```bash
# リモートの変更を取得してマージ
git pull origin main --allow-unrelated-histories
git push -u origin main
```

### 認証エラー
- Personal Access Tokenを再生成
- SSH鍵の設定を確認

