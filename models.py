import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timezone
from typing import List

class AdvancedPredictor:
    def __init__(self):
        self.models = {
            "linear": LinearRegression(),
            "moving_avg": self._create_moving_avg_model()
        }
        
    def _create_moving_avg_model(self):
        class MovingAverage:
            def predict(self, prices, window=3):
                return np.convolve(prices, np.ones(window)/window, mode='valid')[-1]
        return MovingAverage()
    
    def _calculate_confidence(self, prices):
        volatility = np.std(prices[-24:]) / np.mean(prices[-24:])
        trend_strength = np.polyfit(range(len(prices)), prices, 1)[0]
        return max(0.4, min(0.95, 1 - volatility + abs(trend_strength*100)))

    def predict(self, prices: List[float], interval: str) -> dict:
        if len(prices) < 48:
            raise ValueError("Insufficient historical data")
            
        X = np.arange(len(prices)).reshape(-1, 1)
        y = np.array(prices)
        
        self.models["linear"].fit(X, y)
        lr_pred = self.models["linear"].predict([[len(X)]])[0]
        ma_pred = self.models["moving_avg"].predict(y)
        
        return {
            "prediction": round(float((lr_pred * 0.6) + (ma_pred * 0.4)), 4),
            "confidence": round(float(self._calculate_confidence(prices)), 4),
            "interval": interval,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

predictor = AdvancedPredictor()