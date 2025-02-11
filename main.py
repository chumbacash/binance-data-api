from dotenv import load_dotenv
load_dotenv()
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
from services.gemini_insights import GeminiInsightsGenerator
from services.metrics import MetricsTracker


ALLOWED_INTERVALS = ["1h", "4h", "1d", "1w"]
INTERVAL_HOURS = {"1h": 1, "4h": 4, "1d": 24, "1w": 168}
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CryptoPredictAPI")
metrics = MetricsTracker()
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



@app.get("/intraday/{symbol}", tags=["Market Data"])
@limiter.limit("30/minute")
async def get_intraday_data(request: Request, symbol: str):
    """
    Returns intraday data for the given symbol based on 1-hour candles for the current day.
    Data points are fetched from midnight (UTC) until the current time.
    """
    try:
        now = datetime.now(timezone.utc)
        # Determine the start of the current day in UTC
        start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        # Calculate how many full hours have elapsed since the start of the day (plus one to include the current hour)
        hours_elapsed = int((now - start_of_day).total_seconds() // 3600) + 1

        # Fetch hourly OHLCV data for the symbol with limit = hours_elapsed
        hourly_data = binance.fetch_ohlcv(symbol, "1h", limit=hours_elapsed)
        if not hourly_data:
            raise HTTPException(status_code=404, detail="No intraday data available")

        # Filter candles to ensure they are within the current day (if necessary)
        intraday = []
        for candle in hourly_data:
            candle_time = datetime.fromtimestamp(candle["timestamp"] / 1000, tz=timezone.utc)
            if candle_time >= start_of_day:
                intraday.append(candle)

        return {
            "symbol": symbol,
            "intraday_data": intraday,
            "time_updated": now.isoformat(),
            "hours_elapsed": hours_elapsed,
            "candles_returned": len(intraday)
        }
    except Exception as e:
        logger.error(f"Intraday data error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Intraday data retrieval failed")


@app.get("/predict/{symbol}", tags=["Predictions"])
@limiter.limit("30/minute")
async def predict_price(request: Request, symbol: str, interval: str = "1h"):
    try:
        ohlcv = binance.fetch_ohlcv(symbol, interval, limit=100)
        closes = [entry["close"] for entry in ohlcv]
        
        if len(closes) < 50:
            raise HTTPException(
                status_code=422,
                detail="Need at least 50 data points for analysis"
            )
            
        analysis = predictor.analyze_market(closes, interval)
        analysis["metadata"]["symbol"] = symbol
        
        return analysis
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed: " + str(e)
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1))
    )