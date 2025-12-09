"""
config.py のユニットテスト
"""

import pytest
from pydantic import ValidationError


class TestRateLimitSettings:
    """RateLimitSettings のテスト"""

    def test_default_values(self):
        """デフォルト値が正しく設定されること"""
        from config import RateLimitSettings

        settings = RateLimitSettings()
        assert settings.max_retries == 3
        assert settings.base_delay == 1.0
        assert settings.max_delay == 60.0
        assert settings.requests_per_minute == 60
        assert settings.tokens_per_minute == 150000

    def test_custom_values(self, monkeypatch):
        """環境変数からカスタム値を読み込めること"""
        monkeypatch.setenv("RATE_LIMIT_MAX_RETRIES", "5")
        monkeypatch.setenv("RATE_LIMIT_BASE_DELAY", "2.0")

        from config import RateLimitSettings

        settings = RateLimitSettings()
        assert settings.max_retries == 5
        assert settings.base_delay == 2.0


class TestAzureOpenAISettings:
    """AzureOpenAISettings のテスト"""

    def test_endpoint_validation_https(self, monkeypatch):
        """エンドポイントがhttps://で始まることを検証"""
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://invalid.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x" * 32)
        monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "test")

        from config import AzureOpenAISettings

        with pytest.raises(ValidationError) as exc_info:
            AzureOpenAISettings()

        assert "https://" in str(exc_info.value)

    def test_endpoint_trailing_slash_removed(self, monkeypatch):
        """エンドポイントの末尾スラッシュが除去されること"""
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "x" * 32)
        monkeypatch.setenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "test")

        from config import AzureOpenAISettings

        settings = AzureOpenAISettings()
        assert not settings.endpoint.endswith("/")


class TestAzureAISearchSettings:
    """AzureAISearchSettings のテスト"""

    def test_index_name_validation(self, monkeypatch):
        """インデックス名のバリデーション"""
        monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://test.search.windows.net")
        monkeypatch.setenv("AZURE_SEARCH_API_KEY", "x" * 32)
        monkeypatch.setenv("AZURE_SEARCH_INDEX_NAME", "Invalid_Name")

        from config import AzureAISearchSettings

        with pytest.raises(ValidationError):
            AzureAISearchSettings()

    def test_valid_index_name(self, monkeypatch):
        """有効なインデックス名が受け入れられること"""
        monkeypatch.setenv("AZURE_SEARCH_ENDPOINT", "https://test.search.windows.net")
        monkeypatch.setenv("AZURE_SEARCH_API_KEY", "x" * 32)
        monkeypatch.setenv("AZURE_SEARCH_INDEX_NAME", "valid-index-name")

        from config import AzureAISearchSettings

        settings = AzureAISearchSettings()
        assert settings.index_name == "valid-index-name"


class TestValidateConfiguration:
    """validate_configuration のテスト"""

    def test_returns_dict(self):
        """辞書型を返すこと"""
        from config import validate_configuration

        result = validate_configuration()
        assert isinstance(result, dict)
        assert "azure_openai" in result
        assert "azure_search" in result
        assert "rate_limit" in result
