# polygon_driver.py
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd
import requests
from requests import Session
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# -----------------------------------------------------------------------------
# Logging (caller can override level/handlers)
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Minimal default handler; libraries shouldn't configure global logging.
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class PolygonError(Exception):
    """Base error for Polygon driver."""

class PolygonAuthError(PolygonError):
    """Authentication/authorization error (401/403)."""

class PolygonRateLimitError(PolygonError):
    """Rate limit exceeded (429)."""

class PolygonHTTPError(PolygonError):
    """Other HTTP errors."""

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_unix_ms(dt: Union[str, int, float, datetime]) -> int:
    """
    Convert ISO8601 string, datetime, or seconds/ms numeric to Unix ms.
    - Strings: ISO 8601; 'Z' allowed.
    - datetime: timezone-aware preferred; naive assumed UTC.
    - Numbers: interpreted as seconds if < 10^12, else already ms.
    """
    if isinstance(dt, (int, float)):
        # Heuristic: treat < 1e12 as seconds, else milliseconds.
        return int(dt * 1000 if dt < 1_000_000_000_000 else dt)

    if isinstance(dt, str):
        # Support 'Z' suffix
        s = dt.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(s)
        except ValueError as e:
            raise ValueError(f"Invalid datetime string: {dt}") from e
        dt = parsed

    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    raise TypeError("dt must be str | int | float | datetime")

def _ensure_api_key_in_url(url: str, api_key: str) -> str:
    """Append apiKey to a next_url if missing."""
    from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

    parsed = urlparse(url)
    q = dict(parse_qsl(parsed.query))
    if "apiKey" not in q:
        q["apiKey"] = api_key
    new_query = urlencode(q)
    return urlunparse(parsed._replace(query=new_query))

# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
@dataclass
class PolygonTradesDriver:
    api_key: str
    base_url: str = "https://api.polygon.io"
    timeout: float = 15.0
    max_retries: int = 5
    backoff_factor: float = 0.5
    session: Optional[Session] = None

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = self._build_session()

    # -- Session with robust retries ------------------------------------------------
    def _build_session(self) -> Session:
        sess = requests.Session()

        retry = Retry(
            total=self.max_retries,
            connect=self.max_retries,
            read=self.max_retries,
            status=self.max_retries,
            backoff_factor=self.backoff_factor,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
            respect_retry_after_header=True,
        )

        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=20)
        sess.headers.update({"User-Agent": "polygon-driver/1.0"})
        sess.mount("https://", adapter)
        sess.mount("http://", adapter)
        return sess

    # -- Core request wrapper -------------------------------------------------------
    def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        params = dict(params or {})
        # Ensure apiKey is always sent; next_url might omit it
        if "apiKey" not in params and "next_url" not in url:
            params["apiKey"] = self.api_key

        # If url is a next_url from Polygon, make sure apiKey is present in the URL
        if "next_url" in url or url.startswith("https://api.polygon.io/v3/") or "/v3/" in url:
            url = _ensure_api_key_in_url(url, self.api_key)

        resp = self.session.get(url, params=params, timeout=self.timeout)

        if resp.status_code in (401, 403):
            raise PolygonAuthError(f"{resp.status_code} auth error: {resp.text}")
        if resp.status_code == 429:
            raise PolygonRateLimitError("429 rate limit. Consider lowering request pace or upgrading plan.")
        if 400 <= resp.status_code < 600:
            raise PolygonHTTPError(f"{resp.status_code} error: {resp.text}")

        try:
            return resp.json()
        except ValueError as e:
            raise PolygonHTTPError("Invalid JSON response") from e

    def get_trades(
        self,
        ticker: str,
        start: Optional[Union[str, int, float, datetime]] = None,
        end: Optional[Union[str, int, float, datetime]] = None,
        limit: int = 1000,
        sort: str = "asc",
        max_pages: int = 100,
        rate_sleep_s: float = 0.15,
        export_csv: bool = False,
        csv_path: Optional[Union[str, Path]] = None,
        as_dataframe: bool = True,
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Fetch trades from /v3/trades/{ticker} with safe pagination.

        Args:
            ticker: Symbol, e.g., "AAPL".
            start: Inclusive start (ISO string | datetime | seconds/ms).
            end: Inclusive end (ISO string | datetime | seconds/ms).
            limit: Page size (<= 50_000 per Polygon docs; 1000 is safe).
            sort: "asc" | "desc".
            max_pages: Safety cap on pagination.
            rate_sleep_s: Sleep between pages to be gentle with rate limits.
            export_csv: If True, save CSV to csv_path.
            csv_path: Output path (default: "{ticker}_trades.csv").
            as_dataframe: If True, return DataFrame; else list of dicts.

        Returns:
            DataFrame or list of dicts with trades.
        """
        url = f"{self.base_url}/v3/trades/{ticker}"
        params = {
            "limit": limit,
            "order": sort,
            "apiKey": self.api_key,
        }
        if start is not None:
            params["timestamp.gte"] = _to_unix_ms(start)
        if end is not None:
            params["timestamp.lte"] = _to_unix_ms(end)

        all_results: List[Dict] = []
        pages = 0

        try:
            next_url: Optional[str] = url
            while next_url and pages < max_pages:
                # For next_url calls, pass no params; apiKey is ensured in URL
                if next_url != url:
                    data = self._get(next_url, params=None)
                else:
                    data = self._get(next_url, params=params)

                results = data.get("results", [])
                if not isinstance(results, list):
                    logger.warning("Unexpected results payload type: %s", type(results))
                    results = []

                all_results.extend(results)
                next_url = data.get("next_url")
                pages += 1

                if next_url:
                    # Ensure apiKey is present in the next_url
                    next_url = _ensure_api_key_in_url(next_url, self.api_key)
                    time.sleep(rate_sleep_s)

        except PolygonAuthError as e:
            logger.error("Access denied: %s", e)
            return pd.DataFrame() if as_dataframe else []
        except PolygonRateLimitError as e:
            logger.error("Rate limited: %s", e)
            return pd.DataFrame() if as_dataframe else []
        except PolygonHTTPError as e:
            logger.error("HTTP error: %s", e)
            return pd.DataFrame() if as_dataframe else []
        except Exception as e:
            logger.exception("Unexpected error: %s", e)
            return pd.DataFrame() if as_dataframe else []

        if not all_results:
            logger.warning("No trades returned for %s.", ticker)
            return pd.DataFrame() if as_dataframe else []

        if as_dataframe:
            df = pd.DataFrame(all_results)
            for col in ("sip_timestamp", "participant_timestamp", "trf_timestamp", "tape", "sequence_number"):
                if col in df.columns:
                    try:
                        try:
                            df[col] = pd.to_numeric(df[col], errors="coerce") 
                            if df[col].dtype.kind in 'biufc': 
                                df[col] = pd.to_numeric(df[col], downcast="integer")
                        except Exception as e:
                            logger.debug(f"Failed to convert column {col}: {e}")
                    except Exception:
                        pass
            if export_csv:
                path = Path(csv_path or f"{ticker}_trades.csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)
                logger.info("Saved trades to %s", path.as_posix())
            return df
        else:
            if export_csv:
                path = Path(csv_path or f"{ticker}_trades.csv")
                pd.DataFrame(all_results).to_csv(path, index=False)
                logger.info("Saved trades to %s", path.as_posix())
            return all_results

    def get_quotes(
        self,
        ticker: str,
        start: Optional[Union[str, int, float, datetime]] = None,
        end: Optional[Union[str, int, float, datetime]] = None,
        timestamp: Optional[Union[str, int]] = None,
        limit: int = 1000,
        sort: str = "asc",
        order: str = "asc",
        max_pages: int = 100,
        rate_sleep_s: float = 0.15,
        export_csv: bool = False,
        csv_path: Optional[Union[str, Path]] = None,
        as_dataframe: bool = True,
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        Fetch NBBO quotes from /v3/quotes/{ticker} with pagination.

        Parameters
        ----------
        ticker : str
            Case-sensitive symbol, e.g. "AAPL".
        start / end : str | int | float | datetime, optional
            Inclusive range in ISO, datetime, or Unix ms.
        timestamp : str | int, optional
            Exact timestamp (ISO or nanoseconds) – overrides start/end.
        limit : int
            Page size (max 50 000, default 1000).
        sort / order : str
            Sorting direction ("asc" or "desc").
        max_pages : int
            Safety cap on pagination.
        rate_sleep_s : float
            Sleep between pages.
        export_csv : bool
            Save CSV if True.
        csv_path : str | Path, optional
            Output file (default: "{ticker}_quotes.csv").
        as_dataframe : bool
            Return DataFrame (True) or list of dicts (False).

        Returns
        -------
        pd.DataFrame or List[Dict]
        """
        url = f"{self.base_url}/v3/quotes/{ticker}"
        params: dict = {
            "limit": min(limit, 50_000),
            "order": order,
            "sort": sort,
            "apiKey": self.api_key,
        }

        def _to_unix_ms(val) -> int:
            if isinstance(val, (int, float)):
                return int(val * 1_000) if val < 1e12 else int(val)
            if isinstance(val, str):
                return int(pd.to_datetime(val).timestamp() * 1_000)
            if isinstance(val, datetime):
                return int(val.timestamp() * 1_000)
            raise ValueError(f"Cannot convert {val} to Unix ms")

        if timestamp is not None:
            params["timestamp"] = timestamp
        else:
            if start is not None:
                params["timestamp.gte"] = _to_unix_ms(start)
            if end is not None:
                params["timestamp.lte"] = _to_unix_ms(end)

        all_results: List[Dict] = []
        pages = 0
        next_url: Optional[str] = url

        try:
            while next_url and pages < max_pages:
                data = self._get(next_url, params=params if next_url == url else None)
                results = data.get("results", [])
                if not isinstance(results, list):
                    logger.warning("Unexpected results type: %s", type(results))
                    results = []
                all_results.extend(results)

                next_url = data.get("next_url")
                pages += 1
                if next_url:
                    if "apiKey" not in next_url:
                        next_url += f"&apiKey={self.api_key}"
                    time.sleep(rate_sleep_s)
        except Exception as e:
            logger.exception("Failed to fetch quotes for %s: %s", ticker, e)
            return pd.DataFrame() if as_dataframe else []

        if not all_results:
            logger.warning("No quotes returned for %s.", ticker)
            return pd.DataFrame() if as_dataframe else []

        if as_dataframe:
            df = pd.DataFrame(all_results)

            numeric_cols = [
                "ask_exchange", "ask_price", "ask_size",
                "bid_exchange", "bid_price", "bid_size",
                "participant_timestamp", "sip_timestamp",
                "trf_timestamp", "sequence_number", "tape"
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if export_csv:
                path = Path(csv_path or f"{ticker}_quotes.csv")
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(path, index=False)
                logger.info("Saved quotes to %s", path.as_posix())
            return df
        else:
            if export_csv:
                path = Path(csv_path or f"{ticker}_quotes.csv")
                pd.DataFrame(all_results).to_csv(path, index=False)
                logger.info("Saved quotes to %s", path.as_posix())
            return all_results

    def get_last_trade(self, ticker: str) -> Dict:
        """GET /v2/last/trade/{ticker}"""
        url = f"{self.base_url}/v2/last/trade/{ticker}"
        return self._safe_json(url)

    def get_last_quote(self, ticker: str) -> Dict:
        """GET /v2/last/nbbo/{ticker}"""
        url = f"{self.base_url}/v2/last/nbbo/{ticker}"
        return self._safe_json(url)

    def get_daily_bars(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 5000,
        as_dataframe: bool = True,
    ) -> Union[pd.DataFrame, List[Dict]]:
        """
        GET /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
        """
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
        params = {
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": limit,
            "apiKey": self.api_key,
        }
        try:
            data = self._get(url, params=params)
        except PolygonError as e:
            logger.error("Failed to fetch daily bars: %s", e)
            return pd.DataFrame() if as_dataframe else []

        results = data.get("results", []) or []
        return pd.DataFrame(results) if as_dataframe else results

    def get_market_status(self) -> Dict:
        """GET /v1/marketstatus/now"""
        url = f"{self.base_url}/v1/marketstatus/now"
        return self._safe_json(url)

    def get_company_info(self, ticker: str) -> Dict:
        """GET /v3/reference/tickers/{ticker}"""
        url = f"{self.base_url}/v3/reference/tickers/{ticker}"
        return self._safe_json(url)
    
    def get_exchanges(self) -> Dict:
        """GET /v3/reference/exchanges"""
        url = f"{self.base_url}/v3/reference/exchanges"
        return self._safe_json(url)

    def _safe_json(self, url: str, params: Optional[Dict] = None) -> Dict:
        try:
            payload = self._get(url, params=params or {"apiKey": self.api_key})
            return payload if isinstance(payload, dict) else {}
        except PolygonError as e:
            logger.error("Request failed: %s", e)
            return {}

    def close(self) -> None:
        if self.session:
            self.session.close()

    def __enter__(self) -> "PolygonTradesDriver":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
