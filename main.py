from fastapi import FastAPI, HTTPException, status, Request, WebSocket
from datetime import datetime, timezone
import os
import asyncio
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

from data import BinanceClient, SYMBOLS_CACHE, TICKER_CACHE
from models import predictor

ALLOWED_INTERVALS = ["1h", "4h", "1d", "1w"]
INTERVAL_HOURS = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CryptoPredictAPI")

binance = BinanceClient()
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="CryptoPredict Pro+",
    description="Advanced Crypto Analytics & Predictions",
    version="0.3.0",
    docs_url="/docs",
    redoc_url=None
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class LoggingMiddleware:
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        logger.info(f"{request.method} {request.url} - Status: {response.status_code}")
        return response
app.middleware("http")(LoggingMiddleware())

# Endpoints
@app.get("/", tags=["Health"])
@limiter.limit("100/minute")
async def root(request: Request):
    return {
        "name": "CryptoPredict Pro+",
        "status": "online",
        "version": "0.3.0",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/symbols", tags=["Market Data"])
@limiter.limit("30/minute")
async def get_active_symbols(request: Request, sort_by: Optional[str] = None, descending: bool = True):
    try:
        cache_key = "symbols"
        if not SYMBOLS_CACHE.get(cache_key):
            SYMBOLS_CACHE[cache_key] = binance.fetch_symbols()
        symbols = SYMBOLS_CACHE[cache_key]

        if sort_by == "volume":
            sort_cache_key = f"symbols_sorted_volume"
            if not SYMBOLS_CACHE.get(sort_cache_key):
                tickers = binance.fetch_tickers()
                ticker_map = {t['symbol']: t for t in tickers}
                sorted_symbols = sorted(
                    symbols,
                    key=lambda s: float(ticker_map.get(s, {}).get('quoteVolume', 0)),
                    reverse=descending
                )
                SYMBOLS_CACHE[sort_cache_key] = sorted_symbols
            return {
                "symbols": SYMBOLS_CACHE[sort_cache_key],
                "sorting": f"24h_quote_volume_{'desc' if descending else 'asc'}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        return {"symbols": symbols, "timestamp": datetime.now(timezone.utc).isoformat()}
    except Exception as e:
        logger.error(f"Symbols error: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.websocket("/ws/realtime/{symbol}")
async def websocket_realtime(websocket: WebSocket, symbol: str):
    await websocket.accept()
    try:
        while True:
            ohlcv = binance.fetch_ohlcv(symbol, "1h", limit=1)
            if ohlcv:
                await websocket.send_json({
                    "symbol": symbol,
                    "data": ohlcv[0],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            await asyncio.sleep(60)
    except Exception as e:
        logger.error(f"WebSocket error ({symbol}): {str(e)}")
        await websocket.close(code=1011)

@app.get("/predict/{symbol}", tags=["Predictions"])
@limiter.limit("30/minute")
async def predict_price(request: Request, symbol: str, interval: str = "1h"):
    try:
        ohlcv = binance.fetch_ohlcv(symbol, interval, limit=100)
        closes = [entry["close"] for entry in ohlcv]
        return {
            "symbol": symbol,
            "interval": interval,
            "current_price": closes[-1],
            **predictor.predict(closes, interval)
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1))
    )