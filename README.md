# CryptoPredict Pro+ API 🚀

![Docker](https://img.shields.io/badge/Docker-2CA5E0?style=flat&logo=docker&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=FastAPI&logoColor=white)
![Binance](https://img.shields.io/badge/Binance-FCD535?style=flat&logo=binance&logoColor=black)

Advanced cryptocurrency prediction API with real-time analysis and AI-powered insights.

## Features ✨

### Core Features
- 📈 Real-time price predictions with confidence scoring
- 🔍 Technical indicators (SMA, RSI, MACD, Bollinger Bands)
- 🤖 Gemini AI-powered market analysis
- 🌡️ Market health monitoring (volatility, liquidity)

### Technical Features
- ⚡ Async-first architecture for high concurrency
- 🔒 Rate limiting (10 RPM per endpoint)
- 🧠 Smart caching (OHLCV data, predictions)
- 📊 Built-in metrics tracking
- 🛡️ Error resilience with automatic retries

## Environment Setup ⚙️
Create `.env` file with:
```ini
# Required
BINANCE_API=https://api.binance.com/api/v3
GEMINI_API_KEY=your_gemini_key_here
CACHE_TTL=300  # 5 minutes

# Optional
ALLOWED_ORIGINS=*
METRICS_INTERVAL=300  # 5 minutes
API_RATE_LIMIT=100/hour
```

## Docker Deployment 🐳
```bash
# Build and run
docker build -t crypto-api --build-arg GEMINI_API_KEY=$GEMINI_API_KEY .
docker run -p 8000:8000 --env-file .env crypto-api

# Development mode (with hot reload)
docker-compose -f docker-compose.dev.yml up
```

## Project Structure 📁
```text
crypto-data/
├── data-beta/
│   ├── services/               # Business logic components
│   │   ├── gemini_insights.py  # AI analysis integration
│   │   └── metrics.py          # Performance tracking
│   ├── models.py               # Prediction models
│   ├── data.py                 # Binance API client
│   ├── main.py                 # FastAPI entrypoint
│   └── tests/                  # Unit/integration tests
```

## API Endpoints 📡
| Endpoint          | Method | Description                     | Rate Limit   |
|-------------------|--------|---------------------------------|--------------|
| `/predict/{symbol}` | GET    | Price prediction + AI analysis  | 10/min       |
| `/symbols`        | GET    | Active trading pairs            | 30/min       |
| `/intraday/{symbol}` | GET  | Intraday data with custom intervals | 30/min   |
| `/historical/{symbol}` | GET | Historical data with custom intervals | 20/min |
| `/ws/realtime/{symbol}` | WS | Real-time price streaming with custom intervals | - |

## Supported Timeframes ⏰
The API supports the following timeframes for data retrieval and analysis:

- **Hours**: 1h, 2h, 4h, 6h, 8h, 12h
- **Days**: 1d, 3d
- **Weeks**: 1w
- **Months**: 1M

Use these interval values with the `/predict`, `/intraday`, and `/historical` endpoints.

## Rate Limits ⏱️
- Global limit: 100 requests/hour
- Prediction endpoint: 10 requests/minute
- Symbols endpoint: 30 requests/minute
- Exceeding limits returns `429 Too Many Requests`

## Error Handling ❗
Standard error response format:
```json
{
  "error": "Error Type",
  "detail": "Human-readable description",
  "timestamp": "ISO-8601 datetime"
}
```

## Development 🛠️
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload --port 8000

# Run tests
pytest tests/ -v
```

## Testing 🔍
```bash
# Get BTC prediction with 1-hour interval
curl "http://localhost:8000/predict/BTCUSDT?interval=1h"

# Get historical data with 4-hour interval
curl "http://localhost:8000/historical/BTCUSDT?interval=4h&limit=50"

# Get intraday data with 1-hour interval
curl "http://localhost:8000/intraday/BTCUSDT?interval=1h"

# Stream real-time data with 1-hour interval
wscat -c "ws://localhost:8000/ws/realtime/BTCUSDT?interval=1h"
```

## License 📄
MIT License - See [LICENSE](LICENSE) for details

> **Note**  
> This is not financial advice. Cryptocurrency trading carries significant risk.