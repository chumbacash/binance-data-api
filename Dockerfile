# Use official Python image
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    BINANCE_API="https://api.binance.com/api/v3" \
    ALLOWED_ORIGINS="*"

# Expose the port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]