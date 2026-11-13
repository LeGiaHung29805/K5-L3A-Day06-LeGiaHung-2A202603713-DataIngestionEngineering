"""Generate one Gemini vector per chunk or query through the Developer API."""
import json
import logging
import math
import os
from pathlib import Path
import re
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

MODEL = "gemini-embedding-2"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
QUERY_PREFIX = "task: question answering | query: "
DOCUMENT_PREFIX = "title: none | text: "
BATCH_SIZE = 16


class GeminiEmbedder:
    dimensions = 768

    def __init__(self) -> None:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        load_dotenv(env_path, encoding="utf-8-sig")
        self.api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
        if not self.api_key or self.api_key.lower() in {
            "your_key_here", "paste_your_key_here", "your_gemini_api_key",
        }:
            raise ValueError("Set GEMINI_API_KEY in starter/.env (see .env.example).")
        self.model_name = (os.getenv("EMBEDDING_MODEL") or MODEL).strip().removeprefix("models/")
        if self.model_name != MODEL:
            raise ValueError(f"Set EMBEDDING_MODEL={MODEL} in starter/.env.")
        self.config = {
            "provider": "Google Gemini Developer API",
            "model": self.model_name,
            "api_version": "v1beta",
            "dimensions": self.dimensions,
            "document_prefix": DOCUMENT_PREFIX,
            "query_prefix": QUERY_PREFIX,
            "document_batch_size": BATCH_SIZE,
        }
        self.requests = 0
        self.tokens = 0
        self.requests_without_token_count = 0

    def usage_since(self, start: tuple[int, int, int]) -> dict:
        requests, tokens, missing = start
        return {
            "http_requests": self.requests - requests,
            "prompt_tokens_reported": self.tokens - tokens if self.requests_without_token_count == missing else None,
        }

    def usage_snapshot(self) -> tuple[int, int, int]:
        return self.requests, self.tokens, self.requests_without_token_count

    def _request(self, operation: str, body: dict) -> dict:
        request = Request(
            f"{API_ROOT}/models/{self.model_name}:{operation}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        delays = (1, 2, None)
        for delay in delays:
            self.requests += 1
            try:
                with urlopen(request, timeout=30) as response:
                    result = json.load(response)
                count = (result.get("usageMetadata") or {}).get("promptTokenCount")
                if isinstance(count, int):
                    self.tokens += count
                else:
                    self.requests_without_token_count += 1
                return result
            except HTTPError as exc:
                code = exc.code
                details = []
                if code == 429:
                    try:
                        details = json.load(exc).get("error", {}).get("details", [])
                    except (ValueError, OSError):
                        pass
                exc.close()
                if code in {500, 502, 503, 504} and delay is not None:
                    sleep(delay)
                    continue
                if code == 429:
                    quota_ids = [
                        quota_id for item in details for violation in item.get("violations", [])
                        if re.fullmatch(r"[A-Za-z0-9_]{1,100}", quota_id := violation.get("quotaId", ""))
                    ]
                    retry = next((item.get("retryDelay", "") for item in details
                                  if item.get("@type") == "type.googleapis.com/google.rpc.RetryInfo"), "")
                    seconds = float(retry[:-1]) if re.fullmatch(r"\d+(?:\.\d+)?s", retry) else None
                    quota = ", ".join(quota_ids) or "quota not specified"
                    if delay is not None and seconds is not None and seconds <= 60 and not any(
                        "perday" in quota_id.lower() for quota_id in quota_ids
                    ):
                        logging.warning("Gemini HTTP 429 (%s). Retrying in %.1f s.", quota, seconds + 1)
                        sleep(seconds + 1)
                        continue
                    raise RuntimeError(f"Gemini HTTP 429 ({quota}). Check the project's rate limits "
                                       "and billing in Google AI Studio.") from None
                hints = {
                    400: "Check the request and whether the API key is valid.",
                    401: "Check GEMINI_API_KEY in starter/.env.",
                    403: "Check the API key and project access in Google AI Studio.",
                    404: "Check that the project can access gemini-embedding-2.",
                }
                raise RuntimeError(f"Gemini HTTP {code}. {hints.get(code, 'Retry later.')}") from None
            except (URLError, TimeoutError):
                if delay is not None:
                    sleep(delay)
                    continue
                raise RuntimeError("Could not reach Gemini. Check your network connection.") from None
        raise AssertionError("Gemini request retry loop did not return or raise.")

    def _content(self, text: str) -> dict:
        return {
            "model": f"models/{self.model_name}",
            "content": {"parts": [{"text": text}]},
            "outputDimensionality": self.dimensions,
        }

    def _vectors(self, embeddings: list[dict], expected: int) -> list[list[float]]:
        vectors = [embedding.get("values", []) for embedding in embeddings]
        if len(vectors) != expected or any(
            len(vector) != self.dimensions or not any(vector)
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ValueError(f"Expected {expected} nonzero Gemini vectors with {self.dimensions} dimensions.")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), BATCH_SIZE):
            requests = [self._content(DOCUMENT_PREFIX + text) for text in texts[start:start + BATCH_SIZE]]
            result = self._request("batchEmbedContents", {"requests": requests})
            vectors.extend(self._vectors(result.get("embeddings", []), len(requests)))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        result = self._request("embedContent", self._content(QUERY_PREFIX + question))
        return self._vectors([result.get("embedding", {})], 1)[0]
