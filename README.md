# Crypto Prediction API

A FastAPI-based cryptocurrency prediction service with technical analysis and AI insights.

## Features
- Real-time price predictions
- Technical indicators (SMA, RSI, MACD)
- AI-powered market analysis
- Docker support

## Setup
1. Clone the repository
2. Create `.env` file with required secrets
3. Build and run with Docker:
   ```bash
   docker build -t crypto-api .
   docker run -p 8000:8000 crypto-api