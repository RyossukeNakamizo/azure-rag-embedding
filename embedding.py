"""
Azure OpenAI Serviceを使用したテキストベクトル化クラス

text-embedding-3-largeモデルを使用して日本語テキストをベクトル化します。
非同期処理、バッチ処理、リトライロジック、トークン数カウント機能を提供します。
"""

import asyncio
import logging
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
import numpy as np
import tiktoken
from dotenv import load_dotenv
from openai import APIError, APITimeoutError, AsyncAzureOpenAI, RateLimitError

from config import (
    AzureOpenAISettings,
    RateLimitSettings,
    get_azure_openai_settings,
    get_rate_limit_settings,
    get_settings,
)

# .envファイルを読み込む
load_dotenv()

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Embedding生成結果"""

    text: str
    embedding: list[float]
    token_count: int
    latency_ms: float


@dataclass
class BatchEmbeddingResult:
    """バッチEmbedding生成結果"""

    results: list[EmbeddingResult]
    total_tokens: int
    total_latency_ms: float
    failed_indices: list[int] = field(default_factory=list)


@dataclass
class RateLimitState:
    """レート制限状態管理"""

    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    minute_start: float = field(default_factory=time.time)
    _lock: asyncio.Lock | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """asyncio.Lockを初期化"""
        if self._lock is None:
            self._lock = asyncio.Lock()

    def reset_if_needed(self) -> None:
        """1分経過していたらカウンターをリセット"""
        now = time.time()
        if now - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_start = now

    def can_process(self, token_count: int, settings: RateLimitSettings) -> bool:
        """リクエスト可能かどうかを判定"""
        self.reset_if_needed()
        return (
            self.requests_this_minute < settings.requests_per_minute
            and self.tokens_this_minute + token_count <= settings.tokens_per_minute
        )

    def record_request(self, token_count: int) -> None:
        """リクエストを記録"""
        self.requests_this_minute += 1
        self.tokens_this_minute += token_count


@dataclass
class UsageStats:
    """API使用統計情報"""

    total_requests: int = 0
    total_tokens: int = 0
    total_embeddings: int = 0
    rate_limit_errors: int = 0
    api_errors: int = 0
    timeout_errors: int = 0

    def reset(self) -> None:
        """統計情報をリセット"""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_embeddings = 0
        self.rate_limit_errors = 0
        self.api_errors = 0
        self.timeout_errors = 0


class AzureOpenAIEmbedding:
    """
    Azure OpenAI Serviceを使用したテキストベクトル化クラス

    非同期処理、バッチ処理、リトライロジック、トークン数カウント機能を提供します。
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        deployment_name: str | None = None,
        max_batch_size: int | None = None,
        max_retries: int | None = None,
        embedding_dimensions: int | None = None,
        rate_limit_settings: RateLimitSettings | None = None,
    ):
        """
        初期化

        Args:
            endpoint: Azure OpenAIエンドポイントURL（環境変数から読み込み可能）
            api_key: APIキー（環境変数から読み込み可能）
            api_version: APIバージョン（デフォルト: 2024-10-21）
            deployment_name: デプロイメント名（環境変数から読み込み可能）
            max_batch_size: 最大バッチサイズ（デフォルト: 16）
            max_retries: 最大リトライ回数（デフォルト: RateLimitSettingsから取得）
            embedding_dimensions: Embedding次元数（デフォルト: 1536）
            rate_limit_settings: レート制限設定（Noneの場合は環境変数から読み込み）
        """
        # 設定の読み込み（環境変数優先）
        base_settings = get_azure_openai_settings()
        self._rate_limit_settings = rate_limit_settings or get_rate_limit_settings()
        app_settings = get_settings()

        # 引数で指定された値があればそれを使用、なければ環境変数から読み込んだ値を使用
        self.endpoint = endpoint or base_settings.endpoint
        self.api_key = api_key or base_settings.api_key
        self.api_version = api_version or base_settings.api_version
        self.deployment_name = deployment_name or base_settings.embedding_deployment
        self.max_batch_size = max_batch_size or app_settings.batch_size
        self.max_retries = max_retries or self._rate_limit_settings.max_retries
        self.retry_min_wait = self._rate_limit_settings.base_delay
        self.retry_max_wait = self._rate_limit_settings.max_delay
        self.max_tokens_per_request = base_settings.max_tokens_per_request
        self.embedding_dimensions = embedding_dimensions or base_settings.embedding_dimensions

        # Azure OpenAIクライアントの初期化
        endpoint_clean = self.endpoint.rstrip("/")
        self.client = AsyncAzureOpenAI(
            azure_endpoint=endpoint_clean,
            api_key=self.api_key,
            api_version=self.api_version,
        )

        # トークンエンコーダーの初期化（text-embedding-3-large用）
        try:
            # text-embedding-3-largeはcl100k_baseエンコーダーを使用
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"tiktokenエンコーダーの初期化に失敗: {e}")
            try:
                self.encoder = tiktoken.encoding_for_model("text-embedding-3-small")
            except KeyError:
                self.encoder = None

        # レート制限状態管理
        self._rate_limit_state = RateLimitState()

        # 使用統計情報
        self.usage_stats = UsageStats()

        logger.info(
            f"AzureOpenAIEmbedding initialized: "
            f"endpoint={endpoint_clean}, "
            f"deployment={self.deployment_name}"
        )

    def count_tokens(self, text: str) -> int:
        """
        テキストのトークン数をカウント（事前カウント）

        Args:
            text: カウント対象のテキスト

        Returns:
            トークン数
        """
        if self.encoder is None:
            logger.warning("エンコーダーが初期化されていないため、概算値を返します")
            return len(text) // 2

        return len(self.encoder.encode(text))

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        """
        複数テキストのトークン数をカウント

        Args:
            texts: カウント対象のテキストリスト

        Returns:
            トークン数のリスト
        """
        return [self.count_tokens(text) for text in texts]

    async def _wait_for_rate_limit(self, token_count: int) -> None:
        """レート制限に達している場合は待機"""
        while not self._rate_limit_state.can_process(token_count, self._rate_limit_settings):
            wait_time = 60 - (time.time() - self._rate_limit_state.minute_start)
            if wait_time > 0:
                logger.warning(f"レート制限に達しました。{wait_time:.1f}秒待機します")
                await asyncio.sleep(min(wait_time, 5))
            self._rate_limit_state.reset_if_needed()

    def _calculate_backoff_delay(
        self, attempt: int, rate_limit_error: RateLimitError | None = None
    ) -> float:
        """エクスポネンシャルバックオフの待機時間を計算"""
        # RateLimitErrorからretry-afterヘッダーを取得
        if rate_limit_error and hasattr(rate_limit_error, "response"):
            retry_after = rate_limit_error.response.headers.get("retry-after")
            if retry_after:
                return min(float(retry_after), self._rate_limit_settings.max_delay)

        # エクスポネンシャルバックオフ（ジッター付き）
        delay = self._rate_limit_settings.base_delay * (2**attempt)
        jitter = random.uniform(0, 0.1 * delay)
        return min(delay + jitter, self._rate_limit_settings.max_delay)

    async def _embed_with_retry(self, texts: list[str], total_tokens: int) -> list[list[float]]:
        """
        リトライロジック付きでベクトル化を実行

        Args:
            texts: ベクトル化対象のテキストリスト
            total_tokens: 合計トークン数

        Returns:
            ベクトルリスト

        Raises:
            APIError: APIエラーが発生した場合
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                await self._wait_for_rate_limit(total_tokens)

                # API呼び出し
                kwargs = {
                    "model": self.deployment_name,
                    "input": texts,
                }
                if self.embedding_dimensions:
                    kwargs["dimensions"] = self.embedding_dimensions

                response = await self.client.embeddings.create(**kwargs)

                self._rate_limit_state.record_request(total_tokens)

                # レスポンスからEmbeddingを抽出（index順でソート）
                sorted_data = sorted(response.data, key=lambda x: x.index)
                embeddings = [item.embedding for item in sorted_data]

                # 統計情報を更新
                self.usage_stats.total_requests += 1
                self.usage_stats.total_tokens += response.usage.total_tokens
                self.usage_stats.total_embeddings += len(embeddings)

                return embeddings

            except RateLimitError as e:
                last_exception = e
                self.usage_stats.rate_limit_errors += 1
                delay = self._calculate_backoff_delay(attempt, e)
                logger.warning(
                    f"レート制限エラー (試行 {attempt + 1}/{self.max_retries}), "
                    f"{delay:.1f}秒待機: {e}"
                )
                await asyncio.sleep(delay)

            except APITimeoutError as e:
                last_exception = e
                self.usage_stats.timeout_errors += 1
                delay = self._calculate_backoff_delay(attempt)
                logger.warning(
                    f"タイムアウトエラー (試行 {attempt + 1}/{self.max_retries}), "
                    f"{delay:.1f}秒待機: {e}"
                )
                await asyncio.sleep(delay)

            except APIError as e:
                last_exception = e
                # サーバーエラー（5xx）の場合はリトライ
                if e.status_code and e.status_code >= 500:
                    self.usage_stats.api_errors += 1
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"サーバーエラー (試行 {attempt + 1}/{self.max_retries}), "
                        f"{delay:.1f}秒待機: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    # クライアントエラー（4xx）の場合はリトライしない
                    self.usage_stats.api_errors += 1
                    logger.error(f"APIエラー: {e}")
                    raise

        raise last_exception or APIError("最大リトライ回数を超えました")

    async def embed_text(self, text: str) -> list[float]:
        """
        単一テキストをベクトル化（非同期）

        Args:
            text: ベクトル化対象のテキスト

        Returns:
            ベクトル（浮動小数点数のリスト）

        Raises:
            APIError: APIエラーが発生した場合
            ValueError: テキストが空の場合、またはトークン数が上限を超える場合
        """
        if not text or not text.strip():
            raise ValueError("テキストが空です")

        token_count = self.count_tokens(text)
        if token_count > self.max_tokens_per_request:
            raise ValueError(
                f"テキストのトークン数が上限を超えています: "
                f"{token_count} > {self.max_tokens_per_request}"
            )

        embeddings = await self._embed_with_retry([text], token_count)
        return embeddings[0]

    async def embed_single(self, text: str) -> EmbeddingResult:
        """
        単一テキストのEmbeddingを生成（詳細情報付き）

        Args:
            text: 埋め込み対象のテキスト

        Returns:
            EmbeddingResult: 生成結果（レイテンシ情報含む）

        Raises:
            APIError: APIエラーが発生した場合
            ValueError: テキストが空の場合、またはトークン数が上限を超える場合
        """
        if not text or not text.strip():
            raise ValueError("テキストが空です")

        token_count = self.count_tokens(text)
        if token_count > self.max_tokens_per_request:
            raise ValueError(
                f"テキストのトークン数が上限を超えています: "
                f"{token_count} > {self.max_tokens_per_request}"
            )

        start_time = time.perf_counter()
        embeddings = await self._embed_with_retry([text], token_count)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return EmbeddingResult(
            text=text,
            embedding=embeddings[0],
            token_count=token_count,
            latency_ms=latency_ms,
        )

    def _create_batches(
        self,
        texts: Sequence[str],
        token_counts: list[int],
        batch_size: int | None,
    ) -> list[tuple[list[str], list[int]]]:
        """
        トークン制限を考慮してバッチを作成

        Args:
            texts: テキストリスト
            token_counts: 各テキストのトークン数
            batch_size: 最大バッチサイズ

        Returns:
            (texts, token_counts)のタプルのリスト
        """
        effective_batch_size = batch_size or self.max_batch_size

        batches: list[tuple[list[str], list[int]]] = []
        current_texts: list[str] = []
        current_tokens: list[int] = []
        current_token_sum = 0

        for text, tokens in zip(texts, token_counts, strict=True):
            # 単一テキストが制限を超える場合はスキップ
            if tokens > self.max_tokens_per_request:
                logger.warning(
                    f"トークン数が上限を超えるテキストをスキップ: "
                    f"{tokens} > {self.max_tokens_per_request}"
                )
                continue

            # バッチサイズまたはトークン制限に達したら新バッチ
            if (
                len(current_texts) >= effective_batch_size
                or current_token_sum + tokens > self.max_tokens_per_request
            ):
                if current_texts:
                    batches.append((current_texts, current_tokens))
                current_texts = []
                current_tokens = []
                current_token_sum = 0

            current_texts.append(text)
            current_tokens.append(tokens)
            current_token_sum += tokens

        # 残りをバッチに追加
        if current_texts:
            batches.append((current_texts, current_tokens))

        return batches

    async def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        parallel: bool = True,
    ) -> list[list[float]]:
        """
        複数テキストをバッチ処理でベクトル化（非同期）

        Args:
            texts: ベクトル化対象のテキストリスト
            batch_size: バッチサイズ（Noneの場合はデフォルト値を使用）
            parallel: 並列処理を有効にするか（デフォルト: True）

        Returns:
            ベクトルリスト（各テキストに対応するベクトルのリスト）

        Raises:
            APIError: APIエラーが発生した場合
            ValueError: テキストリストが空の場合
        """
        if not texts:
            raise ValueError("テキストリストが空です")

        # トークン数を事前にカウント
        token_counts = self.count_tokens_batch(texts)

        # トークン制限を考慮してバッチを作成
        batches = self._create_batches(texts, token_counts, batch_size)

        if parallel and len(batches) > 1:
            # 並列処理
            tasks = [
                self._embed_with_retry(batch_texts, sum(batch_tokens))
                for batch_texts, batch_tokens in batches
            ]
            results = await asyncio.gather(*tasks)

            # 結果を結合
            all_embeddings = []
            for embeddings in results:
                all_embeddings.extend(embeddings)

            return all_embeddings
        else:
            # 順次処理
            all_embeddings = []
            for batch_texts, batch_tokens in batches:
                embeddings = await self._embed_with_retry(batch_texts, sum(batch_tokens))
                all_embeddings.extend(embeddings)

            return all_embeddings

    async def embed_batch_detailed(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> BatchEmbeddingResult:
        """
        複数テキストのEmbeddingをバッチ生成（詳細情報付き）

        Args:
            texts: 埋め込み対象のテキストリスト
            batch_size: バッチサイズ（Noneの場合はトークン数で自動調整）

        Returns:
            BatchEmbeddingResult: バッチ生成結果
        """
        if not texts:
            return BatchEmbeddingResult(results=[], total_tokens=0, total_latency_ms=0)

        start_time = time.perf_counter()
        token_counts = self.count_tokens_batch(texts)

        # バッチ分割（トークン制限を考慮）
        batches = self._create_batches(texts, token_counts, batch_size)

        results: list[EmbeddingResult] = []
        failed_indices: list[int] = []
        total_tokens = 0
        current_index = 0

        for batch_texts, batch_tokens in batches:
            try:
                embeddings = await self._embed_with_retry(batch_texts, sum(batch_tokens))

                for text, embedding, tokens in zip(batch_texts, embeddings, batch_tokens, strict=True):
                    results.append(
                        EmbeddingResult(
                            text=text,
                            embedding=embedding,
                            token_count=tokens,
                            latency_ms=0,  # バッチ全体のレイテンシのみ記録
                        )
                    )
                    total_tokens += tokens

            except Exception as e:
                logger.error(f"バッチEmbedding生成に失敗: {e}")
                for i in range(len(batch_texts)):
                    failed_indices.append(current_index + i)

            current_index += len(batch_texts)

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        return BatchEmbeddingResult(
            results=results,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            failed_indices=failed_indices,
        )

    async def embed_with_chunking(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        aggregation: str = "mean",
    ) -> EmbeddingResult:
        """
        長文テキストをチャンク分割してEmbeddingを生成

        Args:
            text: 埋め込み対象のテキスト
            chunk_size: チャンクサイズ（トークン数）
            overlap: チャンク間のオーバーラップ（トークン数）
            aggregation: 集約方法（"mean" | "first" | "weighted_mean"）

        Returns:
            EmbeddingResult: 集約されたEmbedding結果

        Raises:
            ValueError: 集約方法が無効な場合
        """
        if self.encoder is None:
            raise ValueError("エンコーダーが初期化されていません")

        tokens = self.encoder.encode(text)
        total_tokens = len(tokens)

        # チャンク分割が不要な場合
        if total_tokens <= self.max_tokens_per_request:
            return await self.embed_single(text)

        # チャンク作成
        chunks: list[str] = []
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunks.append(self.encoder.decode(chunk_tokens))
            start += chunk_size - overlap

        # バッチでEmbedding生成
        batch_result = await self.embed_batch_detailed(chunks)

        if not batch_result.results:
            raise ValueError("すべてのチャンクのEmbedding生成に失敗しました")

        # 集約
        aggregated = self._aggregate_embeddings(
            [r.embedding for r in batch_result.results],
            [r.token_count for r in batch_result.results],
            aggregation,
        )

        return EmbeddingResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            embedding=aggregated,
            token_count=total_tokens,
            latency_ms=batch_result.total_latency_ms,
        )

    def _aggregate_embeddings(
        self,
        embeddings: list[list[float]],
        weights: list[int],
        method: str,
    ) -> list[float]:
        """複数のEmbeddingを集約"""
        arr = np.array(embeddings)

        if method == "first":
            return arr[0].tolist()
        elif method == "mean":
            return np.mean(arr, axis=0).tolist()
        elif method == "weighted_mean":
            weights_arr = np.array(weights, dtype=float)
            weights_arr /= weights_arr.sum()
            return np.average(arr, axis=0, weights=weights_arr).tolist()
        else:
            raise ValueError(f"無効な集約方法: {method}")

    def get_usage_stats(self) -> dict[str, int]:
        """
        API使用統計情報を取得

        Returns:
            使用統計情報の辞書
        """
        return {
            "total_requests": self.usage_stats.total_requests,
            "total_tokens": self.usage_stats.total_tokens,
            "total_embeddings": self.usage_stats.total_embeddings,
            "rate_limit_errors": self.usage_stats.rate_limit_errors,
            "api_errors": self.usage_stats.api_errors,
            "timeout_errors": self.usage_stats.timeout_errors,
        }

    def reset_usage_stats(self) -> None:
        """使用統計情報をリセット"""
        self.usage_stats.reset()

    async def close(self) -> None:
        """リソースをクリーンアップ"""
        await self.client.close()
        logger.info("AzureOpenAIEmbedding closed")

    async def __aenter__(self):
        """非同期コンテキストマネージャーのエントリ"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """非同期コンテキストマネージャーのエグジット"""
        await self.close()


# ============================================================================
# AsyncEmbeddingClient - 新しいクリーンな実装
# ============================================================================


class EmbeddingClientError(Exception):
    """Embeddingクライアント固有のエラー"""

    pass


class MaxRetriesExceededError(EmbeddingClientError):
    """最大リトライ回数超過エラー"""

    pass


@dataclass
class AsyncRateLimitState:
    """レート制限状態管理（asyncio対応）"""

    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    minute_start: float = field(default_factory=time.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def reset_if_needed(self) -> None:
        """1分経過していたらカウンターをリセット"""
        now = time.time()
        if now - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_start = now

    def can_process(self, token_count: int, settings: RateLimitSettings) -> bool:
        """リクエスト可能かどうかを判定"""
        self.reset_if_needed()
        return (
            self.requests_this_minute < settings.requests_per_minute
            and self.tokens_this_minute + token_count <= settings.tokens_per_minute
        )

    def record_request(self, token_count: int) -> None:
        """リクエストを記録"""
        self.requests_this_minute += 1
        self.tokens_this_minute += token_count


class AsyncEmbeddingClient:
    """
    Azure OpenAI 非同期Embeddingクライアント

    Features:
        - 非同期バッチ処理による高スループット
        - エクスポネンシャルバックオフによるリトライ
        - トークン/リクエストベースのレート制限
        - 詳細なメトリクス収集
    """

    def __init__(
        self,
        settings: AzureOpenAISettings | None = None,
        rate_limit_settings: RateLimitSettings | None = None,
        request_timeout: float = 30.0,
    ):
        """
        クライアントを初期化。

        Args:
            settings: Azure OpenAI設定（Noneの場合は環境変数から読み込み）
            rate_limit_settings: レート制限設定（Noneの場合は環境変数から読み込み）
            request_timeout: リクエストタイムアウト（秒）
        """
        self._settings = settings or get_azure_openai_settings()
        self._rate_limit_settings = rate_limit_settings or get_rate_limit_settings()
        self._rate_limit_state = AsyncRateLimitState()

        # タイムアウト設定付きクライアント初期化
        self._client = AsyncAzureOpenAI(
            azure_endpoint=self._settings.endpoint,
            api_key=self._settings.api_key,
            api_version=self._settings.api_version,
            timeout=httpx.Timeout(request_timeout, connect=10.0),
        )

        # tiktokenエンコーダー（トークンカウント用）
        try:
            self._tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
        except KeyError:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")

        # エンドポイントURLをマスク処理（セキュリティ対策）
        masked_endpoint = self._mask_url(self._settings.endpoint)
        logger.info(
            f"AsyncEmbeddingClient initialized: "
            f"endpoint={masked_endpoint}, "
            f"deployment={self._settings.embedding_deployment}"
        )

    @staticmethod
    def _mask_url(url: str) -> str:
        """URLをマスク処理（セキュリティ対策）"""
        try:
            parsed = urlparse(url)
            host_parts = parsed.hostname.split(".")
            if len(host_parts) > 2:
                masked_host = f"{host_parts[0][:3]}***." + ".".join(host_parts[-2:])
            else:
                masked_host = "***." + ".".join(host_parts[-1:])
            return f"{parsed.scheme}://{masked_host}"
        except Exception:
            return "***masked***"

    def count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        return len(self._tokenizer.encode(text))

    def count_tokens_batch(self, texts: Sequence[str]) -> list[int]:
        """複数テキストのトークン数をカウント"""
        return [self.count_tokens(text) for text in texts]

    async def _wait_for_rate_limit(self, token_count: int) -> None:
        """レート制限に達している場合は待機（スレッドセーフ）"""
        async with self._rate_limit_state._lock:
            while not self._rate_limit_state.can_process(token_count, self._rate_limit_settings):
                wait_time = 60 - (time.time() - self._rate_limit_state.minute_start)
                if wait_time > 0:
                    logger.warning(f"レート制限に達しました。{wait_time:.1f}秒待機します")
                    await asyncio.sleep(min(wait_time, 5))
                self._rate_limit_state.reset_if_needed()

    async def _embed_with_retry(self, texts: list[str], total_tokens: int) -> list[list[float]]:
        """
        リトライロジック付きでEmbedding APIを呼び出し。

        Args:
            texts: 埋め込み対象のテキストリスト
            total_tokens: 合計トークン数

        Returns:
            Embeddingベクトルのリスト

        Raises:
            MaxRetriesExceededError: 最大リトライ回数を超えた場合
        """
        last_exception: Exception | None = None
        for attempt in range(self._rate_limit_settings.max_retries):
            try:
                await self._wait_for_rate_limit(total_tokens)
                response = await self._client.embeddings.create(
                    model=self._settings.embedding_deployment,
                    input=texts,
                    dimensions=self._settings.embedding_dimensions,
                )
                async with self._rate_limit_state._lock:
                    self._rate_limit_state.record_request(total_tokens)
                # レスポンスからEmbeddingを抽出（インデックス順でソート）
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]
            except RateLimitError as e:
                last_exception = e
                delay = self._calculate_backoff_delay(attempt, e)
                logger.warning(
                    f"レート制限エラー (試行 {attempt + 1}/{self._rate_limit_settings.max_retries}), "
                    f"{delay:.1f}秒待機: {e}"
                )
                await asyncio.sleep(delay)
            except APITimeoutError as e:
                last_exception = e
                delay = self._calculate_backoff_delay(attempt)
                logger.warning(
                    f"タイムアウトエラー (試行 {attempt + 1}/{self._rate_limit_settings.max_retries}), "
                    f"{delay:.1f}秒待機: {e}"
                )
                await asyncio.sleep(delay)
            except APIError as e:
                last_exception = e
                if e.status_code and e.status_code >= 500:
                    delay = self._calculate_backoff_delay(attempt)
                    logger.warning(
                        f"サーバーエラー (試行 {attempt + 1}/{self._rate_limit_settings.max_retries}), "
                        f"{delay:.1f}秒待機: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        # 適切な例外チェーン
        raise MaxRetriesExceededError(
            f"最大リトライ回数 ({self._rate_limit_settings.max_retries}) を超えました"
        ) from last_exception

    def _calculate_backoff_delay(
        self, attempt: int, rate_limit_error: RateLimitError | None = None
    ) -> float:
        """エクスポネンシャルバックオフの待機時間を計算"""
        # RateLimitErrorからretry-afterヘッダーを取得
        if rate_limit_error and hasattr(rate_limit_error, "response"):
            retry_after = rate_limit_error.response.headers.get("retry-after")
            if retry_after:
                return min(float(retry_after), self._rate_limit_settings.max_delay)

        # エクスポネンシャルバックオフ（ジッター付き）
        delay = self._rate_limit_settings.base_delay * (2**attempt)
        jitter = random.uniform(0, 0.1 * delay)
        return min(delay + jitter, self._rate_limit_settings.max_delay)

    async def embed_single(self, text: str) -> EmbeddingResult:
        """
        単一テキストのEmbeddingを生成。

        Args:
            text: 埋め込み対象のテキスト

        Returns:
            EmbeddingResult: 生成結果
        """
        token_count = self.count_tokens(text)
        if token_count > self._settings.max_tokens_per_request:
            raise ValueError(
                f"テキストのトークン数が上限を超えています: "
                f"{token_count} > {self._settings.max_tokens_per_request}"
            )

        start_time = time.perf_counter()
        embeddings = await self._embed_with_retry([text], token_count)
        latency_ms = (time.perf_counter() - start_time) * 1000

        return EmbeddingResult(
            text=text,
            embedding=embeddings[0],
            token_count=token_count,
            latency_ms=latency_ms,
        )

    async def embed_batch(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> BatchEmbeddingResult:
        """
        複数テキストのEmbeddingをバッチ生成。

        Args:
            texts: 埋め込み対象のテキストリスト
            batch_size: バッチサイズ（Noneの場合はトークン数で自動調整）

        Returns:
            BatchEmbeddingResult: バッチ生成結果
        """
        if not texts:
            return BatchEmbeddingResult(results=[], total_tokens=0, total_latency_ms=0)

        start_time = time.perf_counter()
        token_counts = self.count_tokens_batch(texts)
        # バッチ分割（トークン制限を考慮）
        batches = self._create_batches(texts, token_counts, batch_size)

        results: list[EmbeddingResult] = []
        failed_indices: list[int] = []
        total_tokens = 0
        current_index = 0

        for batch_texts, batch_tokens in batches:
            try:
                embeddings = await self._embed_with_retry(batch_texts, sum(batch_tokens))
                for _, (text, embedding, tokens) in enumerate(
                    zip(batch_texts, embeddings, batch_tokens, strict=True)
                ):
                    results.append(
                        EmbeddingResult(
                            text=text,
                            embedding=embedding,
                            token_count=tokens,
                            latency_ms=0,  # バッチ全体のレイテンシのみ記録
                        )
                    )
                    total_tokens += tokens
            except Exception as e:
                logger.error(f"バッチEmbedding生成に失敗: {e}")
                for i in range(len(batch_texts)):
                    failed_indices.append(current_index + i)
            current_index += len(batch_texts)

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        return BatchEmbeddingResult(
            results=results,
            total_tokens=total_tokens,
            total_latency_ms=total_latency_ms,
            failed_indices=failed_indices,
        )

    def _create_batches(
        self,
        texts: Sequence[str],
        token_counts: list[int],
        batch_size: int | None,
    ) -> list[tuple[list[str], list[int]]]:
        """
        トークン制限を考慮してバッチを作成。

        Args:
            texts: テキストリスト
            token_counts: 各テキストのトークン数
            batch_size: 最大バッチサイズ

        Returns:
            (texts, token_counts)のタプルのリスト
        """
        max_tokens = self._settings.max_tokens_per_request
        effective_batch_size = batch_size or 16

        batches: list[tuple[list[str], list[int]]] = []
        current_texts: list[str] = []
        current_tokens: list[int] = []
        current_token_sum = 0

        for text, tokens in zip(texts, token_counts, strict=True):
            # 単一テキストが制限を超える場合はスキップ（エラーログ出力）
            if tokens > max_tokens:
                logger.warning(
                    f"トークン数が上限を超えるテキストをスキップ: {tokens} > {max_tokens}"
                )
                continue

            # バッチサイズまたはトークン制限に達したら新バッチ
            if (
                len(current_texts) >= effective_batch_size
                or current_token_sum + tokens > max_tokens
            ):
                if current_texts:
                    batches.append((current_texts, current_tokens))
                current_texts = []
                current_tokens = []
                current_token_sum = 0

            current_texts.append(text)
            current_tokens.append(tokens)
            current_token_sum += tokens

        # 残りをバッチに追加
        if current_texts:
            batches.append((current_texts, current_tokens))

        return batches

    async def embed_with_chunking(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        aggregation: str = "mean",
    ) -> EmbeddingResult:
        """
        長文テキストをチャンク分割してEmbeddingを生成。

        Args:
            text: 埋め込み対象のテキスト
            chunk_size: チャンクサイズ（トークン数）
            overlap: チャンク間のオーバーラップ（トークン数）
            aggregation: 集約方法（"mean" | "first" | "weighted_mean"）

        Returns:
            EmbeddingResult: 集約されたEmbedding結果
        """
        tokens = self._tokenizer.encode(text)
        total_tokens = len(tokens)

        # チャンク分割が不要な場合
        if total_tokens <= self._settings.max_tokens_per_request:
            return await self.embed_single(text)

        # チャンク作成
        chunks: list[str] = []
        start = 0
        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunks.append(self._tokenizer.decode(chunk_tokens))
            start += chunk_size - overlap

        # バッチでEmbedding生成
        batch_result = await self.embed_batch(chunks)

        if not batch_result.results:
            raise ValueError("すべてのチャンクのEmbedding生成に失敗しました")

        # 集約
        aggregated = self._aggregate_embeddings(
            [r.embedding for r in batch_result.results],
            [r.token_count for r in batch_result.results],
            aggregation,
        )

        return EmbeddingResult(
            text=text[:100] + "..." if len(text) > 100 else text,
            embedding=aggregated,
            token_count=total_tokens,
            latency_ms=batch_result.total_latency_ms,
        )

    def _aggregate_embeddings(
        self,
        embeddings: list[list[float]],
        weights: list[int],
        method: str,
    ) -> list[float]:
        """複数のEmbeddingを集約"""
        arr = np.array(embeddings)

        if method == "first":
            return arr[0].tolist()
        elif method == "mean":
            return np.mean(arr, axis=0).tolist()
        elif method == "weighted_mean":
            weights_arr = np.array(weights, dtype=float)
            weights_arr /= weights_arr.sum()
            return np.average(arr, axis=0, weights=weights_arr).tolist()
        else:
            raise ValueError(f"無効な集約方法: {method}")

    async def close(self) -> None:
        """クライアントをクローズ"""
        await self._client.close()
        logger.info("AsyncEmbeddingClient closed")

    async def __aenter__(self) -> "AsyncEmbeddingClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
