# Crypto Prediction API:

A FastAPI-based cryptocurrency prediction service with technical analysis and AI insights.

## Features:
- Real-time price predictions
- Technical indicators (SMA, RSI, MACD)
- AI-powered market analysis
- Docker support

## Setup:
1. Clone the repository
2. Create `.env` file with required secrets
3. Build and run with Docker:
   ```bash
   docker build -t crypto-api .
   docker run -p 8000:8000 crypto-api
   ```

## Project structure:
crypto-data/
└── data-beta/
    ├── .dockerignore          # Docker ignore file
    ├── Dockerfile             # Docker configuration
    ├── docker-compose.yml     # Optional for local development
    ├── requirements.txt       # Python dependencies
    ├── .env                   # Environment variables (DO NOT COMMIT THIS)
    ├── .gitignore             # Git ignore file
    ├── main.py                # Your FastAPI app
    ├── models.py              # Prediction models
    ├── services/              # Additional services
    │   └── gemini_insights.py
    └── README.md              # Project documentation