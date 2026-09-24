"""Transport hardening for an internal service: request-size limit and opt-in CORS."""
import logging
import os
from dataclasses import dataclass
from typing import Tuple

from flask import Flask
from flask_cors import CORS

logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH_ENV = "MAX_CONTENT_LENGTH_BYTES"
CORS_ORIGINS_ENV = "CORS_ORIGINS"
FLASK_DEBUG_ENV = "FLASK_DEBUG"
MEBIBYTE = 1024 * 1024
TRUTHY = frozenset({"true", "1", "yes"})


@dataclass(frozen=True)
class HttpPolicy:
    """Request-size limit and CORS policy, built from the environment.

    - ``MAX_CONTENT_LENGTH_BYTES``: oversized bodies are rejected with 413 before
      they are read into memory.
    - ``CORS_ORIGINS``: comma-separated browser origins. Empty (default) means no
      CORS headers: the service is only called server-to-server.
    """

    max_content_length: int
    cors_origins: Tuple[str, ...] = ()

    @classmethod
    def from_environment(cls, default_max_bytes: int) -> "HttpPolicy":
        max_bytes = int(os.getenv(MAX_CONTENT_LENGTH_ENV, str(default_max_bytes)))
        if max_bytes <= 0:
            raise ValueError(f"{MAX_CONTENT_LENGTH_ENV} must be positive, got {max_bytes}")
        raw = os.getenv(CORS_ORIGINS_ENV, "")
        origins = tuple(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip())
        return cls(max_bytes, origins)

    def apply(self, app: Flask) -> None:
        app.config["MAX_CONTENT_LENGTH"] = self.max_content_length
        if self.cors_origins:
            CORS(app, origins=list(self.cors_origins), supports_credentials=False)
            logger.info(f"CORS origins: {', '.join(self.cors_origins)}")

    @staticmethod
    def debug_enabled() -> bool:
        """Flask debug (Werkzeug debugger = remote code execution) is opt-in only."""
        return os.getenv(FLASK_DEBUG_ENV, "false").strip().lower() in TRUTHY
