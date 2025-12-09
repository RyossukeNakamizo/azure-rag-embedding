"""
Azure OpenAI & AI Search 環境変数管理モジュール

Pydantic Settingsを使用した型安全な設定管理。
.envファイルまたは環境変数から設定を読み込み、バリデーションを実行。
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureOpenAISettings(BaseSettings):
    """Azure OpenAI サービス設定"""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str = Field(
        ...,
        description="Azure OpenAI エンドポイントURL",
        examples=["https://your-resource.openai.azure.com/"],
    )
    api_key: str = Field(
        ...,
        description="Azure OpenAI APIキー",
        min_length=20,  # Fixed: Azure API key length varies
    )
    api_version: str = Field(
        default="2024-10-21",
        description="Azure OpenAI APIバージョン",
    )
    # Fixed: Removed redundant alias - env_prefix already handles AZURE_OPENAI_ prefix
    embedding_deployment: str = Field(
        ...,
        description="Embeddingモデルのデプロイメント名",
        examples=["text-embedding-ada-002", "text-embedding-3-small"],
    )
    embedding_dimensions: int = Field(
        default=1536,
        ge=256,
        le=3072,
        description="Embeddingベクトルの次元数",
    )
    max_tokens_per_request: int = Field(
        default=8191,
        ge=1,
        le=8191,
        description="1リクエストあたりの最大トークン数",
    )

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """エンドポイントURLの正規化とバリデーション"""
        if not v.startswith("https://"):
            raise ValueError("エンドポイントはhttps://で始まる必要があります")
        return v.rstrip("/")


class AzureAISearchSettings(BaseSettings):
    """Azure AI Search サービス設定"""

    model_config = SettingsConfigDict(
        env_prefix="AZURE_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    endpoint: str = Field(
        ...,
        description="Azure AI Search エンドポイントURL",
        examples=["https://your-search.search.windows.net"],
    )
    api_key: str = Field(
        ...,
        description="Azure AI Search 管理キーまたはクエリキー",
        min_length=20,  # Fixed: API key length varies
    )
    index_name: str = Field(
        ...,
        description="検索インデックス名",
        min_length=1,
        max_length=128,
    )
    semantic_config_name: Optional[str] = Field(
        default=None,
        description="セマンティック検索設定名（オプション）",
    )
    api_version: str = Field(
        default="2024-07-01",
        description="Azure AI Search APIバージョン",
    )

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, v: str) -> str:
        """エンドポイントURLの正規化"""
        if not v.startswith("https://"):
            raise ValueError("エンドポイントはhttps://で始まる必要があります")
        return v.rstrip("/")

    @field_validator("index_name")
    @classmethod
    def validate_index_name(cls, v: str) -> str:
        """インデックス名のバリデーション（Azure命名規則準拠）"""
        import re

        if not re.match(r"^[a-z0-9][a-z0-9-]*$", v):
            raise ValueError(
                "インデックス名は小文字英数字で始まり、小文字英数字とハイフンのみ使用可能です"
            )
        return v


class RateLimitSettings(BaseSettings):
    """レート制限設定"""

    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    max_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="最大リトライ回数",
    )
    base_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="リトライ基本待機時間（秒）",
    )
    max_delay: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="リトライ最大待機時間（秒）",
    )
    requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="1分あたりの最大リクエスト数",
    )
    tokens_per_minute: int = Field(
        default=150000,
        ge=1000,
        le=10000000,
        description="1分あたりの最大トークン数",
    )


class Settings(BaseSettings):
    """統合設定クラス"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Lazy initialization to avoid multiple .env reads
    _azure_openai: Optional[AzureOpenAISettings] = None
    _azure_search: Optional[AzureAISearchSettings] = None
    _rate_limit: Optional[RateLimitSettings] = None

    # アプリケーション設定
    log_level: str = Field(
        default="INFO",
        description="ログレベル",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        le=100,
        description="Embeddingバッチサイズ",
    )
    request_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="リクエストタイムアウト（秒）",
    )

    @property
    def azure_openai(self) -> AzureOpenAISettings:
        """Azure OpenAI設定を遅延初期化で取得"""
        if self._azure_openai is None:
            self._azure_openai = AzureOpenAISettings()
        return self._azure_openai

    @property
    def azure_search(self) -> AzureAISearchSettings:
        """Azure AI Search設定を遅延初期化で取得"""
        if self._azure_search is None:
            self._azure_search = AzureAISearchSettings()
        return self._azure_search

    @property
    def rate_limit(self) -> RateLimitSettings:
        """レート制限設定を遅延初期化で取得"""
        if self._rate_limit is None:
            self._rate_limit = RateLimitSettings()
        return self._rate_limit


@lru_cache()
def get_settings() -> Settings:
    """
    設定のシングルトンインスタンスを取得。

    Returns:
        Settings: 環境変数から読み込まれた設定インスタンス

    Raises:
        ValidationError: 必須環境変数が未設定または不正な値の場合
    """
    return Settings()


def get_azure_openai_settings() -> AzureOpenAISettings:
    """Azure OpenAI設定を取得"""
    return get_settings().azure_openai


def get_azure_search_settings() -> AzureAISearchSettings:
    """Azure AI Search設定を取得"""
    return get_settings().azure_search


def get_rate_limit_settings() -> RateLimitSettings:
    """レート制限設定を取得"""
    return get_settings().rate_limit


# 設定検証用ユーティリティ
def validate_configuration() -> dict[str, bool]:
    """
    全設定の検証を実行し、結果を返す。

    Returns:
        dict: 各設定項目の検証結果
    """
    results = {
        "azure_openai": False,
        "azure_search": False,
        "rate_limit": False,
    }

    try:
        _ = AzureOpenAISettings()
        results["azure_openai"] = True
    except Exception:
        pass

    try:
        _ = AzureAISearchSettings()
        results["azure_search"] = True
    except Exception:
        pass

    try:
        _ = RateLimitSettings()
        results["rate_limit"] = True
    except Exception:
        pass

    return results


if __name__ == "__main__":
    # 設定検証テスト
    print("=== 設定検証 ===")
    validation_results = validate_configuration()
    for key, valid in validation_results.items():
        status = "✓" if valid else "✗"
        print(f"{status} {key}: {'OK' if valid else 'FAILED'}")

