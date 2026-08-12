"""Resilient API client used by the Streamlit views."""

from __future__ import annotations

from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import UISettings, ui_settings


class APIError(RuntimeError):
    def __init__(
        self, message: str, *, status_code: int | None = None, request_id: str | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id

    @property
    def user_message(self) -> str:
        suffix = f" (request {self.request_id})" if self.request_id else ""
        return f"{self}{suffix}"


class ESGAPIClient:
    def __init__(self, config: UISettings = ui_settings, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()
        retry = Retry(
            total=config.api_retry_count,
            connect=config.api_retry_count,
            read=config.api_retry_count,
            status=config.api_retry_count,
            backoff_factor=0.4,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST", "DELETE"}),
            respect_retry_after_header=True,
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _headers(self, endpoint: str, *, stream: bool = False) -> dict[str, str]:
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        if "/admin/" in endpoint and self.config.admin_token:
            headers["Authorization"] = f"Bearer {self.config.admin_token.get_secret_value()}"
        return headers

    def request(
        self,
        endpoint: str,
        *,
        method: str = "GET",
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.config.esg_api_url}/{endpoint.lstrip('/')}"
        timeout = (
            self.config.api_connect_timeout_seconds,
            self.config.api_read_timeout_seconds,
        )
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=None if files else data,
                data=data if files else None,
                files=files,
                headers=self._headers(endpoint),
                timeout=timeout,
            )
            self._raise_for_status(response)
            return response.json() if response.content else None
        except APIError:
            raise
        except requests.RequestException as exc:
            raise APIError(f"Cannot reach the API at {self.config.esg_api_url}") from exc

    def stream(self, endpoint: str, data: dict[str, Any] | None = None) -> requests.Response:
        url = f"{self.config.esg_api_url}/{endpoint.lstrip('/')}"
        try:
            response = self.session.post(
                url,
                json=data,
                stream=True,
                headers=self._headers(endpoint, stream=True),
                timeout=(
                    self.config.api_connect_timeout_seconds,
                    self.config.api_read_timeout_seconds,
                ),
            )
            self._raise_for_status(response)
            return response
        except APIError:
            raise
        except requests.RequestException as exc:
            raise APIError(f"Cannot reach the API at {self.config.esg_api_url}") from exc

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.ok:
            return
        message = f"API request failed with status {response.status_code}"
        request_id = response.headers.get("X-Request-ID")
        try:
            payload = response.json()
            detail = payload.get("detail")
            if isinstance(detail, dict):
                message = str(detail.get("detail") or detail.get("error") or message)
            elif detail:
                message = str(detail)
            else:
                message = str(payload.get("error") or message)
            request_id = payload.get("request_id") or request_id
        except ValueError:
            pass
        raise APIError(message, status_code=response.status_code, request_id=request_id)


api_client = ESGAPIClient()
