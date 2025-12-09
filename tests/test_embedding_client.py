"""
embedding_client.py のユニットテスト
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestEmbeddingResult:
    """EmbeddingResult のテスト"""

    def test_dataclass_creation(self):
        """データクラスが正しく作成されること"""
        from embedding_client import EmbeddingResult

        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2, 0.3],
            token_count=5,
            latency_ms=100.0,
        )
        assert result.text == "test"
        assert len(result.embedding) == 3
        assert result.token_count == 5


class TestBatchEmbeddingResult:
    """BatchEmbeddingResult のテスト"""

    def test_default_failed_indices(self):
        """failed_indicesのデフォルトが空リストであること"""
        from embedding_client import BatchEmbeddingResult, EmbeddingResult

        result = BatchEmbeddingResult(
            results=[],
            total_tokens=0,
            total_latency_ms=0,
        )
        assert result.failed_indices == []


class TestRateLimitState:
    """RateLimitState のテスト"""

    def test_initial_state(self):
        """初期状態が正しいこと"""
        from embedding_client import RateLimitState

        state = RateLimitState()
        assert state.requests_this_minute == 0
        assert state.tokens_this_minute == 0

    def test_record_request(self):
        """リクエスト記録が正しく動作すること"""
        from embedding_client import RateLimitState

        state = RateLimitState()
        state.record_request(100)
        assert state.requests_this_minute == 1
        assert state.tokens_this_minute == 100

    def test_can_process_within_limits(self):
        """制限内でcan_processがTrueを返すこと"""
        from embedding_client import RateLimitState
        from config import RateLimitSettings

        state = RateLimitState()
        settings = RateLimitSettings()
        assert state.can_process(100, settings) is True

    def test_has_asyncio_lock(self):
        """asyncio.Lockを持っていること"""
        from embedding_client import RateLimitState
        import asyncio

        state = RateLimitState()
        assert hasattr(state, "_lock")
        assert isinstance(state._lock, asyncio.Lock)


class TestAsyncEmbeddingClient:
    """AsyncEmbeddingClient のテスト"""

    @pytest.fixture
    def mock_tiktoken(self):
        """tiktokenのモック"""
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = list(range(10))
        mock_encoding.decode.return_value = "decoded"

        with patch("embedding_client.tiktoken") as mock:
            mock.encoding_for_model.return_value = mock_encoding
            mock.get_encoding.return_value = mock_encoding
            yield mock

    def test_mask_url(self, mock_tiktoken):
        """URLマスクが正しく動作すること"""
        from embedding_client import AsyncEmbeddingClient

        masked = AsyncEmbeddingClient._mask_url(
            "https://my-resource.openai.azure.com/"
        )
        assert "my-resource" not in masked or "***" in masked

    def test_count_tokens(self, mock_tiktoken):
        """トークンカウントが動作すること"""
        from embedding_client import AsyncEmbeddingClient

        client = AsyncEmbeddingClient()
        count = client.count_tokens("test text")
        assert isinstance(count, int)

    def test_calculate_backoff_delay(self, mock_tiktoken):
        """バックオフ計算が指数関数的に増加すること"""
        from embedding_client import AsyncEmbeddingClient

        client = AsyncEmbeddingClient()

        delay0 = client._calculate_backoff_delay(0)
        delay1 = client._calculate_backoff_delay(1)
        delay2 = client._calculate_backoff_delay(2)

        # 指数関数的に増加（ジッターがあるので大まかに確認）
        assert delay1 > delay0
        assert delay2 > delay1

    def test_aggregate_embeddings_mean(self, mock_tiktoken):
        """mean集約が正しく計算されること"""
        from embedding_client import AsyncEmbeddingClient

        client = AsyncEmbeddingClient()
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        result = client._aggregate_embeddings(embeddings, [1, 1], "mean")

        assert result == [2.0, 3.0]

    def test_aggregate_embeddings_first(self, mock_tiktoken):
        """first集約が最初の要素を返すこと"""
        from embedding_client import AsyncEmbeddingClient

        client = AsyncEmbeddingClient()
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        result = client._aggregate_embeddings(embeddings, [1, 1], "first")

        assert result == [1.0, 2.0]

    def test_aggregate_embeddings_invalid_method(self, mock_tiktoken):
        """無効な集約方法でエラーが発生すること"""
        from embedding_client import AsyncEmbeddingClient

        client = AsyncEmbeddingClient()

        with pytest.raises(ValueError, match="Unknown aggregation method"):
            client._aggregate_embeddings([[1.0]], [1], "invalid")

    def test_create_batches(self, mock_tiktoken):
        """バッチ作成が正しく動作すること"""
        from embedding_client import AsyncEmbeddingClient

        client = AsyncEmbeddingClient()
        texts = ["a", "b", "c", "d", "e"]
        token_counts = [10, 10, 10, 10, 10]

        batches = client._create_batches(texts, token_counts, batch_size=2)

        assert len(batches) == 3  # 2, 2, 1
        assert len(batches[0][0]) == 2
        assert len(batches[2][0]) == 1


class TestCustomExceptions:
    """カスタム例外のテスト"""

    def test_exception_hierarchy(self):
        """例外階層が正しいこと"""
        from embedding_client import (
            EmbeddingClientError,
            MaxRetriesExceededError,
        )

        assert issubclass(MaxRetriesExceededError, EmbeddingClientError)
        assert issubclass(EmbeddingClientError, Exception)

    def test_max_retries_error_message(self):
        """MaxRetriesExceededErrorのメッセージが正しいこと"""
        from embedding_client import MaxRetriesExceededError

        error = MaxRetriesExceededError("Test message")
        assert str(error) == "Test message"
