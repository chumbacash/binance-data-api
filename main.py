from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException, Request, WebSocket
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
from fastapi.staticfiles import StaticFiles
from data import BinanceClient, SYMBOLS_CACHE, TICKER_CACHE
from models import predictor
from services.gemini_insights import GeminiInsightsGenerator
from services.metrics import MetricsTracker

ALLOWED_INTERVALS = ["1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
INTERVAL_HOURS = {
    "1h": 1, 
    "2h": 2, 
    "4h": 4, 
    "6h": 6, 
    "8h": 8, 
    "12h": 12, 
    "1d": 24, 
    "3d": 72, 
    "1w": 168, 
    "1M": 720
}
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CoolifyCryptoAPI")
metrics = MetricsTracker()
binance = BinanceClient()
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Coolify CryptoPredict Pro+",
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
app.mount("/static", StaticFiles(directory="static"), name="static")


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
async def websocket_realtime(websocket: WebSocket, symbol: str, interval: str = "1h"):
    await websocket.accept()
    try:
        # Validate interval
        if interval not in ALLOWED_INTERVALS:
            await websocket.send_json({
                "error": "Invalid interval",
                "detail": f"Allowed values: {', '.join(ALLOWED_INTERVALS)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            await websocket.close(code=1008)
            return
            
        while True:
            # Await the async fetch_ohlcv call here
            ohlcv = await binance.fetch_ohlcv(symbol, interval, limit=1)
            if ohlcv:
                await websocket.send_json({
                    "symbol": symbol,
                    "interval": interval,
                    "data": ohlcv[0],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            
            # Adjust sleep time based on interval
            if interval.endswith('h'):
                # For hour intervals, sleep for 5 minutes
                sleep_time = 300
            elif interval in ['1d', '3d', '1w', '1M']:
                # For day/week/month intervals, sleep for 15 minutes
                sleep_time = 900
            else:
                sleep_time = 300
                
            await asyncio.sleep(sleep_time)
    except Exception as e:
        logger.error(f"WebSocket error ({symbol}): {str(e)}")
        await websocket.close(code=1011)

@app.get("/intraday/{symbol}", tags=["Market Data"])
@limiter.limit("30/minute")
async def get_intraday_data(request: Request, symbol: str, interval: str = "1h"):
    """
    Returns intraday data for the given symbol based on the specified interval for the current day.
    Data points are fetched from midnight (UTC) until the current time.
    """
    try:
        # Validate interval
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
            
        now = datetime.now(timezone.utc)
        # Determine the start of the current day in UTC
        start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        
        # Calculate how many intervals have elapsed since the start of the day
        # For intervals less than 1 hour, we need more data points
        interval_hours = INTERVAL_HOURS[interval]
        intervals_elapsed = int((now - start_of_day).total_seconds() / (interval_hours * 3600)) + 1
        
        # Limit to a reasonable number of candles
        limit = min(intervals_elapsed, 500)
        
        # Await the async call for OHLCV data
        data = await binance.fetch_ohlcv(symbol, interval, limit=limit)
        if not data:
            raise HTTPException(status_code=404, detail="No intraday data available")

        # Filter candles to ensure they are within the current day (if necessary)
        intraday = []
        for candle in data:
            candle_time = datetime.fromtimestamp(candle["timestamp"] / 1000, tz=timezone.utc)
            if candle_time >= start_of_day:
                intraday.append(candle)

        return {
            "symbol": symbol,
            "interval": interval,
            "intraday_data": intraday,
            "time_updated": now.isoformat(),
            "intervals_elapsed": intervals_elapsed,
            "candles_returned": len(intraday)
        }
    except Exception as e:
        logger.error(f"Intraday data error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Intraday data retrieval failed")

@app.get("/predict/{symbol}", tags=["Predictions"])
@limiter.limit("30/minute")
async def predict_price(request: Request, symbol: str, interval: str = "1h"):
    try:
        # Debug logging for interval
        logger.info(f"Predict endpoint called with interval: {interval}")
        
        # Validate interval
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
            
        # Clear the cache to ensure fresh analysis
        predictor.analysis_cache.clear()
        logger.info(f"Cache cleared for fresh analysis with interval: {interval}")
            
        # Await the async call for OHLCV data
        ohlcv = await binance.fetch_ohlcv(symbol, interval, limit=100)
        closes = [entry["close"] for entry in ohlcv]
        
        if len(closes) < 50:
            raise HTTPException(
                status_code=422,
                detail="Need at least 50 data points for analysis"
            )
            
        # Await the async analyze_market call
        analysis = await predictor.analyze_market(closes, interval)
        
        # Force the interval in the response
        analysis["metadata"]["interval"] = interval
        analysis["metadata"]["symbol"] = symbol
        
        return analysis
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Analysis failed: " + str(e)
        )

@app.get("/historical/{symbol}", tags=["Market Data"])
@limiter.limit("20/minute")
async def get_historical_data(
    request: Request, 
    symbol: str, 
    interval: str = "1h", 
    limit: int = 100
):
    """
    Returns historical data for the given symbol and interval.
    Allows specifying the number of candles to retrieve.
    """
    try:
        # Validate interval
        if interval not in ALLOWED_INTERVALS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interval. Allowed values: {', '.join(ALLOWED_INTERVALS)}"
            )
            
        # Validate limit
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=400,
                detail="Limit must be between 1 and 1000"
            )
            
        # Await the async call for OHLCV data
        data = await binance.fetch_ohlcv(symbol, interval, limit=limit)
        if not data:
            raise HTTPException(status_code=404, detail="No historical data available")

        return {
            "symbol": symbol,
            "interval": interval,
            "historical_data": data,
            "time_updated": datetime.now(timezone.utc).isoformat(),
            "candles_returned": len(data)
        }
    except Exception as e:
        logger.error(f"Historical data error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Historical data retrieval failed")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=int(os.getenv("WORKERS", 1))
    )
