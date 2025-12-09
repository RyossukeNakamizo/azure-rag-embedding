# Azure OpenAI Embedding

Azure OpenAI Serviceの`text-embedding-3-large`モデルを使用して日本語テキストをベクトル化するPythonクラスです。

## 機能

- ✅ 非同期処理対応（asyncio）
- ✅ バッチ処理（最大16テキスト/リクエスト）
- ✅ リトライロジック（exponential backoff）
- ✅ トークン数カウント機能（事前カウント + APIレスポンス）
- ✅ 環境変数からの設定読み込み
- ✅ 型ヒント完備
- ✅ エラーハンドリング（RateLimitError, APIError）

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. 環境変数の設定

`.env.example`をコピーして`.env`ファイルを作成し、Azure OpenAI Serviceの設定を入力してください。

```bash
cp .env.example .env
```

`.env`ファイルに以下の情報を設定します：

```env
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=text-embedding-3-large
AZURE_OPENAI_MAX_BATCH_SIZE=16
```

### 環境変数の説明

- `AZURE_OPENAI_ENDPOINT`: Azure OpenAIエンドポイントURL（必須）
- `AZURE_OPENAI_API_KEY`: APIキー（必須）
- `AZURE_OPENAI_API_VERSION`: APIバージョン（デフォルト: `2024-02-15-preview`）
- `AZURE_OPENAI_DEPLOYMENT_NAME`: デプロイメント名（デフォルト: `text-embedding-3-large`）
- `AZURE_OPENAI_MAX_BATCH_SIZE`: バッチサイズ（デフォルト: `16`）

## 使用方法

### 基本的な使用例

```python
import asyncio
from embedding import AzureOpenAIEmbedding


async def main():
    # インスタンスの作成（環境変数から自動的に設定を読み込み）
    embedding = AzureOpenAIEmbedding()
    
    # 単一テキストのベクトル化
    text = "こんにちは、世界！"
    vector = await embedding.embed_text(text)
    print(f"ベクトル次元数: {len(vector)}")
    
    # トークン数のカウント
    token_count = embedding.count_tokens(text)
    print(f"トークン数: {token_count}")
    
    # リソースのクリーンアップ
    await embedding.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### バッチ処理の例

```python
import asyncio
from embedding import AzureOpenAIEmbedding


async def main():
    embedding = AzureOpenAIEmbedding()
    
    # 複数テキストのバッチ処理
    texts = [
        "これは最初のテキストです。",
        "これは二番目のテキストです。",
        "これは三番目のテキストです。",
    ]
    
    # 並列処理でベクトル化（デフォルト）
    vectors = await embedding.embed_batch(texts, parallel=True)
    
    for i, vector in enumerate(vectors):
        print(f"テキスト{i+1}のベクトル次元数: {len(vector)}")
    
    # 使用統計情報の取得
    stats = embedding.get_usage_stats()
    print(f"総リクエスト数: {stats['total_requests']}")
    print(f"総トークン数: {stats['total_tokens']}")
    print(f"総ベクトル数: {stats['total_embeddings']}")
    
    await embedding.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### コンテキストマネージャーの使用

```python
import asyncio
from embedding import AzureOpenAIEmbedding


async def main():
    # コンテキストマネージャーを使用すると自動的にリソースがクリーンアップされます
    async with AzureOpenAIEmbedding() as embedding:
        text = "コンテキストマネージャーの例"
        vector = await embedding.embed_text(text)
        print(f"ベクトル次元数: {len(vector)}")


if __name__ == "__main__":
    asyncio.run(main())
```

### カスタム設定での初期化

```python
import asyncio
from embedding import AzureOpenAIEmbedding


async def main():
    # カスタム設定で初期化
    embedding = AzureOpenAIEmbedding(
        endpoint="https://custom-endpoint.openai.azure.com/",
        api_key="custom-api-key",
        deployment_name="custom-deployment",
        max_batch_size=8,
        max_retries=3,
    )
    
    text = "カスタム設定の例"
    vector = await embedding.embed_text(text)
    
    await embedding.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## エラーハンドリング

### RateLimitError（レート制限エラー）

レート制限エラーが発生した場合、自動的にexponential backoffでリトライします。

```python
import asyncio
from embedding import AzureOpenAIEmbedding
from openai import RateLimitError


async def main():
    embedding = AzureOpenAIEmbedding()
    
    try:
        text = "レート制限のテスト"
        vector = await embedding.embed_text(text)
    except RateLimitError as e:
        print(f"レート制限エラー: {e}")
        # 自動的にリトライされますが、最大リトライ回数を超えた場合は例外が発生します
    
    await embedding.close()


if __name__ == "__main__":
    asyncio.run(main())
```

### APIError（一般的なAPIエラー）

```python
import asyncio
from embedding import AzureOpenAIEmbedding
from openai import APIError


async def main():
    embedding = AzureOpenAIEmbedding()
    
    try:
        text = "APIエラーのテスト"
        vector = await embedding.embed_text(text)
    except APIError as e:
        print(f"APIエラー: {e}")
        # 一時的なエラーの場合は自動的にリトライされます
    
    await embedding.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## APIリファレンス

### AzureOpenAIEmbedding

#### メソッド

##### `embed_text(text: str) -> List[float]`

単一テキストをベクトル化します。

- **引数**:
  - `text`: ベクトル化対象のテキスト
- **戻り値**: ベクトル（浮動小数点数のリスト）
- **例外**: `ValueError`（テキストが空の場合）、`APIError`（APIエラー）

##### `embed_batch(texts: List[str], parallel: bool = True) -> List[List[float]]`

複数テキストをバッチ処理でベクトル化します。

- **引数**:
  - `texts`: ベクトル化対象のテキストリスト
  - `parallel`: 並列処理を有効にするか（デフォルト: `True`）
- **戻り値**: ベクトルリスト（各テキストに対応するベクトルのリスト）
- **例外**: `ValueError`（テキストリストが空の場合）、`APIError`（APIエラー）

##### `count_tokens(text: str) -> int`

テキストのトークン数をカウントします（事前カウント）。

- **引数**:
  - `text`: カウント対象のテキスト
- **戻り値**: トークン数

##### `get_usage_stats() -> Dict[str, int]`

API使用統計情報を取得します。

- **戻り値**: 使用統計情報の辞書
  - `total_requests`: 総リクエスト数
  - `total_tokens`: 総トークン数
  - `total_embeddings`: 総ベクトル数
  - `rate_limit_errors`: レート制限エラー数
  - `api_errors`: APIエラー数

##### `reset_usage_stats() -> None`

使用統計情報をリセットします。

##### `close() -> None`

リソースをクリーンアップします。

## リトライロジック

デフォルトでは、以下の設定でリトライが行われます：

- **最大リトライ回数**: 5回
- **待機時間**: Exponential backoff（最小1秒、最大60秒）
- **リトライ対象**: `RateLimitError`、`APIError`

リトライ設定は初期化時に変更できます：

```python
embedding = AzureOpenAIEmbedding(
    max_retries=10,  # 最大リトライ回数を10回に変更
)
```

## 注意事項

- Azure OpenAI ServiceのエンドポイントとAPIキーが必要です
- バッチサイズは最大16テキスト/リクエストが推奨されています
- トークン数カウントは`tiktoken`ライブラリを使用していますが、APIレスポンスからも取得可能です
- 非同期処理を使用するため、`asyncio.run()`または`await`を使用してください

## ライセンス

このプロジェクトのライセンス情報は各依存ライブラリのライセンスに従います。

