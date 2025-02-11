import os
import requests
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from typing import List, Dict
import asyncio

logger = logging.getLogger("CryptoPredictAPI")

BINANCE_API = os.getenv("BINANCE_API", "https://api.binance.com/api/v3")
CACHE_TTL = int(os.getenv("CACHE_TTL", 120))

# Caching setup
SYMBOLS_CACHE = TTLCache(maxsize=2, ttl=CACHE_TTL)
OHLCV_CACHE = TTLCache(maxsize=1000, ttl=300)
TICKER_CACHE = TTLCache(maxsize=1, ttl=300)
CACHE_LOCK = asyncio.Lock()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

class BinanceClient:
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(requests.HTTPError)
    )
    def fetch_symbols(self) -> List[str]:
        try:
            response = requests.get(f"{BINANCE_API}/exchangeInfo", headers=HEADERS)
            response.raise_for_status()
            return [s["symbol"] for s in response.json()["symbols"] if s["status"] == "TRADING"]
        except Exception as e:
            logger.error(f"Symbols fetch error: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.HTTPError, requests.Timeout))
    )
    async def fetch_ohlcv(self, symbol: str, interval: str = "1h", limit: int = 100) -> List[Dict]:
        async with CACHE_LOCK:
            cache_key = f"{symbol}_{interval}"
            if cache_key in OHLCV_CACHE:
                return OHLCV_CACHE[cache_key]
            
            try:
                response = await asyncio.to_thread(
                    requests.get,
                    f"{BINANCE_API}/klines?symbol={symbol}&interval={interval}&limit={limit}",
                    headers=HEADERS,
                    timeout=5
                )
                response.raise_for_status()
                data = response.json()
                
                ohlcv = [{
                    "timestamp": entry[0],
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                } for entry in data]
                
                OHLCV_CACHE[cache_key] = ohlcv
                return ohlcv
            except Exception as e:
                logger.error(f"OHLCV fetch error: {str(e)}")
                raise

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type(requests.HTTPError)
    )
    def fetch_tickers(self) -> List[dict]:
        try:
            if 'tickers' in TICKER_CACHE:
                return TICKER_CACHE['tickers']
                
            response = requests.get(f"{BINANCE_API}/ticker/24hr", headers=HEADERS)
            response.raise_for_status()
            tickers = response.json()
            TICKER_CACHE['tickers'] = tickers
            return tickers
        except Exception as e:
            logger.error(f"Tickers fetch error: {str(e)}")
            raise