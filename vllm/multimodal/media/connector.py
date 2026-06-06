# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import atexit
import contextlib
import hashlib
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar
from urllib.request import url2pathname

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Histogram
from urllib3.util import Url, parse_url

import vllm.envs as envs
from vllm.connections import HTTPConnection, global_http_connection
from vllm.logger import init_logger
from vllm.utils.registry import ExtensionManager

from .audio import AudioEmbeddingMediaIO, AudioMediaIO
from .base import MediaIO
from .image import ImageEmbeddingMediaIO, ImageMediaIO
from .video import VideoMediaIO

logger = init_logger(__name__)


def _build_placeholder_image_bytes() -> bytes:
    """Encode a 1x1 blank (white) RGB image as PNG bytes.

    Used as a fallback when ``VLLM_MEDIA_LOADING_BEST_EFFORT`` is enabled and an
    image fails to download or decode. Substituting a valid placeholder (instead
    of dropping the item) keeps the number of multimodal placeholders matched
    with the number of images, so the request can still be processed by the
    downstream multimodal processor and chat templates (e.g. qwen-vl).
    """
    with BytesIO() as buffer:
        Image.new("RGB", (1, 1), (255, 255, 255)).save(buffer, format="PNG")
        return buffer.getvalue()


# Encoded once at import time; decoding back to a PIL image is cheap.
_PLACEHOLDER_IMAGE_BYTES = _build_placeholder_image_bytes()


# ---------------------------------------------------------------------------
# Prometheus metrics for image fetching. Exposed on the same /metrics endpoint
# as the rest of vLLM (``vllm:`` prefix, scrapeable by Grafana). They are
# registered in the API server frontend process, which is where MediaConnector
# (and thus the image download) runs.
#   domain : image source host (or "data" / "file" / "unknown")
#   status : success | error
# Failure rate per domain can be derived from vllm:mm_image_fetch_total as
#   error / (success + error).
# ---------------------------------------------------------------------------
# NOTE: These metrics are created lazily (on first observation) rather than at
# import time. ``vllm.v1.metrics.prometheus.unregister_vllm_metrics()`` (called
# from ``PrometheusStatLogger.__init__``) unregisters *every* collector whose
# name contains "vllm" from the global REGISTRY during engine startup. Since
# these metrics are named ``vllm:mm_image_fetch_*``, registering them at import
# time would get them silently removed before any request is served, so they
# would never appear on ``/metrics``. Creating them lazily ensures registration
# happens after that cleanup has run.
_IMAGE_FETCH_DURATION: Histogram | None = None
_IMAGE_FETCH_TOTAL: Counter | None = None


def _get_image_fetch_metrics() -> tuple[Histogram, Counter]:
    """Lazily create and return the (duration, total) image-fetch metrics."""
    global _IMAGE_FETCH_DURATION, _IMAGE_FETCH_TOTAL
    if _IMAGE_FETCH_DURATION is None or _IMAGE_FETCH_TOTAL is None:
        _IMAGE_FETCH_DURATION = Histogram(
            "vllm:mm_image_fetch_duration_seconds",
            "Duration of fetching an image (download + decode) via "
            "MediaConnector, labeled by source domain and status.",
            labelnames=("domain", "status"),
            buckets=(
                0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0
            ),
        )
        _IMAGE_FETCH_TOTAL = Counter(
            "vllm:mm_image_fetch_total",
            "Total image fetch attempts via MediaConnector, split by source "
            "domain and status (success/error).",
            labelnames=("domain", "status"),
        )
    return _IMAGE_FETCH_DURATION, _IMAGE_FETCH_TOTAL


def _image_fetch_domain(url: str) -> str:
    """Best-effort extraction of the source domain for metric labels."""
    if url.startswith("data:"):
        return "data"
    try:
        url_spec = parse_url(url)
    except Exception:
        return "unknown"
    if url_spec.scheme == "file":
        return "file"
    return url_spec.hostname or "unknown"


def _observe_image_fetch(domain: str, elapsed: float, status: str) -> None:
    """Record latency and success/error count for a single image fetch."""
    duration, total = _get_image_fetch_metrics()
    total.labels(domain, status).inc()
    duration.labels(domain, status).observe(elapsed)


_M = TypeVar("_M")

global_thread_pool = ThreadPoolExecutor(
    max_workers=envs.VLLM_MEDIA_LOADING_THREAD_COUNT
)
atexit.register(global_thread_pool.shutdown)

MEDIA_CONNECTOR_REGISTRY = ExtensionManager()

MODALITY_IO_MAP: dict[str, type[MediaIO]] = {
    "audio": AudioMediaIO,
    "image": ImageMediaIO,
    "video": VideoMediaIO,
}


def merge_media_io_kwargs(
    defaults: dict[str, dict[str, Any]] | None,
    overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Merge config-level and per-request media_io_kwargs per modality.

    Each modality key is merged using the corresponding MediaIO subclass's
    ``merge_kwargs``, which may apply modality-specific logic (e.g.
    VideoMediaIO clears cross-dependent fps/num_frames fields).
    """
    if not defaults and not overrides:
        return None
    all_keys = set(defaults or {}) | set(overrides or {})
    merged = {}
    for key in all_keys:
        io_cls = MODALITY_IO_MAP.get(key, MediaIO)
        merged[key] = io_cls.merge_kwargs(
            (defaults or {}).get(key),
            (overrides or {}).get(key),
        )
    return merged or None


@MEDIA_CONNECTOR_REGISTRY.register("http")
class MediaConnector:
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(
        self,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        connection: HTTPConnection = global_http_connection,
        *,
        allowed_local_media_path: str = "",
        allowed_media_domains: list[str] | None = None,
    ) -> None:
        """
        Args:
            media_io_kwargs: Additional args passed to process media
                             inputs, keyed by modalities. For example,
                             to set num_frames for video, set
                             `--media-io-kwargs '{"video":{"num_frames":40}}'`
            connection: HTTP connection client to download media contents.
            allowed_local_media_path: A local directory to load media files from.
            allowed_media_domains: If set, only media URLs that belong to this
                                   domain can be used for multi-modal inputs.
        """
        super().__init__()

        self.media_io_kwargs: dict[str, dict[str, Any]] = (
            media_io_kwargs if media_io_kwargs else {}
        )
        self.connection = connection

        if allowed_local_media_path:
            allowed_local_media_path_ = Path(allowed_local_media_path)

            if not allowed_local_media_path_.exists():
                raise ValueError(
                    "Invalid `--allowed-local-media-path`: The path "
                    f"{allowed_local_media_path_} does not exist."
                )
            if not allowed_local_media_path_.is_dir():
                raise ValueError(
                    "Invalid `--allowed-local-media-path`: The path "
                    f"{allowed_local_media_path_} must be a directory."
                )
        else:
            allowed_local_media_path_ = None

        self.allowed_local_media_path = allowed_local_media_path_
        if allowed_media_domains is None:
            allowed_media_domains = []
        self.allowed_media_domains = allowed_media_domains

        # Media download cache (opt-in via VLLM_MEDIA_CACHE)
        self._media_cache_dir: str | None = None
        self._media_cache_max_bytes: int = 0
        self._media_cache_ttl_secs: float = 0
        media_cache = envs.VLLM_MEDIA_CACHE
        if media_cache:
            try:
                os.makedirs(media_cache, exist_ok=True)
                # Verify the directory is writable before enabling caching
                with tempfile.NamedTemporaryFile(dir=media_cache, delete=True):
                    pass
                self._media_cache_dir = media_cache
                self._media_cache_max_bytes = (
                    envs.VLLM_MEDIA_CACHE_MAX_SIZE_MB * 1024 * 1024
                )
                self._media_cache_ttl_secs = envs.VLLM_MEDIA_CACHE_TTL_HOURS * 3600
                logger.info(
                    "Media cache enabled at %s (max %d MB, TTL %s hours)",
                    media_cache,
                    envs.VLLM_MEDIA_CACHE_MAX_SIZE_MB,
                    envs.VLLM_MEDIA_CACHE_TTL_HOURS,
                )
            except OSError:
                logger.warning(
                    "VLLM_MEDIA_CACHE path %s is not writable, media caching disabled",
                    media_cache,
                )

    def _get_cached_bytes(self, url: str) -> bytes | None:
        """Return cached bytes for a URL, or None if not cached/expired."""
        if not self._media_cache_dir:
            return None
        cache_path = self._media_cache_path(url)
        # Check TTL
        try:
            age = time.time() - cache_path.stat().st_mtime
        except OSError:
            return None
        if age > self._media_cache_ttl_secs:
            cache_path.unlink(missing_ok=True)
            return None
        # Touch mtime for LRU ordering
        try:
            cache_path.touch()
            return cache_path.read_bytes()
        except OSError:
            return None

    def _put_cached_bytes(self, url: str, data: bytes) -> None:
        """Store downloaded bytes and evict if over budget."""
        if not self._media_cache_dir:
            return
        cache_path = self._media_cache_path(url)
        # Atomic write via temp file + rename
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self._media_cache_dir, delete=False
            ) as tmp_file:
                tmp_file.write(data)
                tmp_path = tmp_file.name
            os.rename(tmp_path, str(cache_path))
        except OSError:
            # Another process beat us or disk issue
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)
            return
        self._maybe_evict(exclude=cache_path)

    def _maybe_evict(self, exclude: Path | None = None) -> None:
        """Evict expired entries first, then LRU until under size limit."""
        cache_dir = Path(self._media_cache_dir)  # type: ignore[arg-type]
        entries = []
        expired = []
        total_size = 0
        now = time.time()
        for f in cache_dir.iterdir():
            if f.name.startswith("."):
                continue
            try:
                stat = f.stat()
            except OSError:
                continue
            age = now - stat.st_mtime
            if age > self._media_cache_ttl_secs:
                expired.append(f)
                continue
            total_size += stat.st_size
            # Never evict the file we just wrote
            if exclude is not None and f.name == exclude.name:
                continue
            entries.append((stat.st_mtime, stat.st_size, f))

        # Evict items according to LRU policy
        entries.sort(key=lambda e: e[0], reverse=True)
        while total_size > self._media_cache_max_bytes and entries:
            mtime, size, f = entries.pop()
            expired.append(f)
            total_size -= size

        for f in expired:
            f.unlink(missing_ok=True)

    def _media_cache_path(self, url: str) -> Path:
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:20]
        ext = Path(url.split("?", 1)[0]).suffix or ""
        return Path(self._media_cache_dir) / f"{url_hash}{ext}"  # type: ignore[arg-type]

    def _load_data_url(
        self,
        url_spec: Url,
        media_io: MediaIO[_M],
    ) -> _M:  # type: ignore[type-var]
        url_spec_path = url_spec.path or ""
        data_spec, data = url_spec_path.split(",", 1)
        media_type, data_type = data_spec.split(";", 1)
        # media_type starts with a leading "/" (e.g., "/video/jpeg")
        media_type = media_type.lstrip("/")

        if data_type != "base64":
            msg = "Only base64 data URLs are supported for now."
            raise NotImplementedError(msg)

        return media_io.load_base64(media_type, data)

    def _load_file_url(
        self,
        url_spec: Url,
        media_io: MediaIO[_M],
    ) -> _M:  # type: ignore[type-var]
        allowed_local_media_path = self.allowed_local_media_path
        if allowed_local_media_path is None:
            raise RuntimeError(
                "Cannot load local files without `--allowed-local-media-path`."
            )

        url_spec_path = url_spec.path or ""
        url_spec_netloc = url_spec.netloc or ""
        filepath = Path(url2pathname(url_spec_netloc + url_spec_path))
        if allowed_local_media_path not in filepath.resolve().parents:
            raise ValueError(
                f"The file path {filepath} must be a subpath "
                f"of `--allowed-local-media-path {allowed_local_media_path}`."
            )

        return media_io.load_file(filepath)

    def _assert_url_in_allowed_media_domains(self, url_spec: Url) -> None:
        if (
            self.allowed_media_domains
            and url_spec.hostname not in self.allowed_media_domains
        ):
            raise ValueError(
                f"The URL must be from one of the allowed domains: "
                f"{self.allowed_media_domains}. Input URL domain: "
                f"{url_spec.hostname}"
            )

    def load_from_url(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> _M:  # type: ignore[type-var]
        url_spec = parse_url(url)

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            self._assert_url_in_allowed_media_domains(url_spec)

            cached = self._get_cached_bytes(url)
            if cached is not None:
                return media_io.load_bytes(cached)

            connection = self.connection
            data = connection.get_bytes(
                url_spec.url,
                timeout=fetch_timeout,
                allow_redirects=envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS,
            )

            self._put_cached_bytes(url, data)
            return media_io.load_bytes(data)

        if url_spec.scheme == "data":
            return self._load_data_url(url_spec, media_io)

        if url_spec.scheme == "file":
            return self._load_file_url(url_spec, media_io)

        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)

    async def load_from_url_async(
        self,
        url: str,
        media_io: MediaIO[_M],
        *,
        fetch_timeout: int | None = None,
    ) -> _M:
        url_spec = parse_url(url)
        loop = asyncio.get_running_loop()

        if url_spec.scheme and url_spec.scheme.startswith("http"):
            self._assert_url_in_allowed_media_domains(url_spec)

            cached = await loop.run_in_executor(
                global_thread_pool, self._get_cached_bytes, url
            )
            if cached is not None:
                future = loop.run_in_executor(
                    global_thread_pool, media_io.load_bytes, cached
                )
                return await future

            connection = self.connection
            data = await connection.async_get_bytes(
                url_spec.url,
                timeout=fetch_timeout,
                allow_redirects=envs.VLLM_MEDIA_URL_ALLOW_REDIRECTS,
            )

            await loop.run_in_executor(
                global_thread_pool, self._put_cached_bytes, url, data
            )
            future = loop.run_in_executor(global_thread_pool, media_io.load_bytes, data)
            return await future

        if url_spec.scheme == "data":
            future = loop.run_in_executor(
                global_thread_pool, self._load_data_url, url_spec, media_io
            )
            return await future

        if url_spec.scheme == "file":
            future = loop.run_in_executor(
                global_thread_pool, self._load_file_url, url_spec, media_io
            )
            return await future
        msg = "The URL must be either a HTTP, data or file URL."
        raise ValueError(msg)

    def fetch_audio(
        self,
        audio_url: str,
    ) -> tuple[np.ndarray, int | float]:
        """
        Load audio from a URL.
        """
        audio_io = AudioMediaIO(**self.media_io_kwargs.get("audio", {}))

        return self.load_from_url(
            audio_url,
            audio_io,
            fetch_timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )

    async def fetch_audio_async(
        self,
        audio_url: str,
    ) -> tuple[np.ndarray, int | float]:
        """
        Asynchronously fetch audio from a URL.
        """
        audio_io = AudioMediaIO(**self.media_io_kwargs.get("audio", {}))

        return await self.load_from_url_async(
            audio_url,
            audio_io,
            fetch_timeout=envs.VLLM_AUDIO_FETCH_TIMEOUT,
        )

    def fetch_image(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Image.Image:
        """
        Load a PIL image from an HTTP or base64 data URL.

        By default, the image is converted into RGB format.

        Emits ``vllm:mm_image_fetch_*`` metrics (latency + success/error count)
        labeled by source domain. When ``VLLM_MEDIA_LOADING_BEST_EFFORT`` is
        enabled, a download/decode failure returns a 1x1 blank placeholder image
        instead of raising, so the request can continue.
        """
        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )

        domain = _image_fetch_domain(image_url)
        start = time.perf_counter()
        status = "success"
        try:
            return self.load_from_url(
                image_url,
                image_io,
                fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
            )
        except UnidentifiedImageError as e:
            status = "error"
            if envs.VLLM_MEDIA_LOADING_BEST_EFFORT:
                return self._placeholder_image(image_url, image_io, e)
            # convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e
        except Exception as e:
            status = "error"
            if envs.VLLM_MEDIA_LOADING_BEST_EFFORT:
                return self._placeholder_image(image_url, image_io, e)
            raise
        finally:
            _observe_image_fetch(domain, time.perf_counter() - start, status)

    async def fetch_image_async(
        self,
        image_url: str,
        *,
        image_mode: str = "RGB",
    ) -> Image.Image:
        """
        Asynchronously load a PIL image from an HTTP or base64 data URL.

        By default, the image is converted into RGB format.

        See :meth:`fetch_image` for the emitted metrics and the best-effort
        placeholder behavior controlled by ``VLLM_MEDIA_LOADING_BEST_EFFORT``.
        """
        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )

        domain = _image_fetch_domain(image_url)
        start = time.perf_counter()
        status = "success"
        try:
            return await self.load_from_url_async(
                image_url,
                image_io,
                fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
            )
        except UnidentifiedImageError as e:
            status = "error"
            if envs.VLLM_MEDIA_LOADING_BEST_EFFORT:
                return self._placeholder_image(image_url, image_io, e)
            # convert to ValueError to be properly caught upstream
            raise ValueError(str(e)) from e
        except Exception as e:
            status = "error"
            if envs.VLLM_MEDIA_LOADING_BEST_EFFORT:
                return self._placeholder_image(image_url, image_io, e)
            raise
        finally:
            _observe_image_fetch(domain, time.perf_counter() - start, status)

    def _placeholder_image(
        self,
        image_url: str,
        image_io: ImageMediaIO,
        exc: Exception,
    ) -> Image.Image:
        """Return a 1x1 blank placeholder image for a failed image load.

        Only used when ``VLLM_MEDIA_LOADING_BEST_EFFORT`` is enabled. Returning a
        valid placeholder (instead of dropping the item) keeps the number of
        multimodal placeholders matched with the number of images, so the
        request can continue without breaking chat-template placeholder
        validation (e.g. qwen-vl).
        """
        url_preview = image_url if len(image_url) <= 100 else f"{image_url[:100]}..."
        logger.warning(
            "Failed to fetch image from %s; VLLM_MEDIA_LOADING_BEST_EFFORT is "
            "enabled, substituting a 1x1 blank placeholder image so the request "
            "can continue: %s",
            url_preview,
            exc,
        )
        return image_io.load_bytes(_PLACEHOLDER_IMAGE_BYTES)

    def fetch_video(
        self,
        video_url: str,
        *,
        image_mode: str = "RGB",
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Load video from an HTTP or base64 data URL.
        """
        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )
        video_io = VideoMediaIO(image_io, **self.media_io_kwargs.get("video", {}))

        return self.load_from_url(
            video_url,
            video_io,
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    async def fetch_video_async(
        self,
        video_url: str,
        *,
        image_mode: str = "RGB",
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        """
        Asynchronously load video from an HTTP or base64 data URL.

        By default, the image is converted into RGB format.
        """
        image_io = ImageMediaIO(
            image_mode=image_mode, **self.media_io_kwargs.get("image", {})
        )
        video_io = VideoMediaIO(image_io, **self.media_io_kwargs.get("video", {}))

        return await self.load_from_url_async(
            video_url,
            video_io,
            fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
        )

    def fetch_image_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """
        Load image embedding from a URL.
        """
        image_embedding_io = ImageEmbeddingMediaIO()

        return image_embedding_io.load_base64("", data)

    def fetch_audio_embedding(
        self,
        data: str,
    ) -> torch.Tensor:
        """
        Load audio embedding from a URL.
        """
        audio_embedding_io = AudioEmbeddingMediaIO()

        return audio_embedding_io.load_base64("", data)
