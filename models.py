# models.py
import numpy as np
from fastapi import HTTPException 
from datetime import datetime, timezone
from typing import List, Dict
import logging
from services.gemini_insights import GeminiInsightsGenerator
from functools import lru_cache
import asyncio
from cachetools import TTLCache

logger = logging.getLogger("CryptoPredictAPI")

class TechnicalAnalyzer:
    def __init__(self, prices: List[float]):
        self.prices = prices
        
    def calculate_volatility(self, window: int = 24) -> float:
        """Calculate price volatility as standard deviation"""
        returns = np.diff(self.prices[-window:]) / self.prices[-window:-1]
        return float(np.std(returns))

    def find_key_levels(self) -> Dict:
        """Identify support/resistance using recent price extremes"""
        lookback = min(100, len(self.prices))
        recent_highs = [h for h in self.prices[-lookback:] if h >= np.percentile(self.prices[-lookback:], 70)]
        recent_lows = [l for l in self.prices[-lookback:] if l <= np.percentile(self.prices[-lookback:], 30)]
        
        return {
            "support": float(np.mean(recent_lows)) if recent_lows else None,
            "resistance": float(np.mean(recent_highs)) if recent_highs else None
        }

    def generate_messages(self, indicators: Dict) -> Dict:
        """Create human-readable insights from technical data"""
        messages = {
            "summary": "",
            "key_insights": [],
            "action_guide": {}
        }

        # Trend analysis
        if indicators['sma_20'] > indicators['sma_50']:
            messages['key_insights'].append("Bullish SMA crossover (20 > 50)")
        else:
            messages['key_insights'].append("Bearish SMA crossover (20 < 50)")

        # RSI analysis
        if indicators['rsi'] > 70:
            messages['key_insights'].append("Overbought (RSI > 70)")
        elif indicators['rsi'] < 30:
            messages['key_insights'].append("Oversold (RSI < 30)")

        # MACD analysis
        if indicators['macd']['histogram'] > 0:
            messages['key_insights'].append("Bullish MACD momentum")
        else:
            messages['key_insights'].append("Bearish MACD momentum")

        # Generate summary
        bull_count = sum(1 for insight in messages['key_insights'] if "Bullish" in insight)
        bear_count = sum(1 for insight in messages['key_insights'] if "Bearish" in insight)
        
        if bull_count > bear_count:
            messages['summary'] = "Bullish Bias Detected"
            messages['action_guide'] = {"buy": 0.7, "sell": 0.3, "hold": 0.5}
        elif bear_count > bull_count:
            messages['summary'] = "Bearish Bias Detected"
            messages['action_guide'] = {"buy": 0.3, "sell": 0.7, "hold": 0.4}
        else:
            messages['summary'] = "Neutral Market Conditions"
            messages['action_guide'] = {"buy": 0.5, "sell": 0.5, "hold": 0.6}

        return messages

class AdvancedPredictor:
    def __init__(self):
        self.gemini = GeminiInsightsGenerator()
        self.analysis_cache = TTLCache(maxsize=500, ttl=300)  # Add prediction cache
        self.prices = []
        self.indicators = {
            "sma_20": None,
            "sma_50": None,
            "rsi": None,
            "macd": {}
        }
        
    def _calculate_sma(self, window: int) -> float:
        return float(np.mean(self.prices[-window:]))

    def _calculate_rsi(self, window: int = 14) -> float:
        deltas = np.diff(self.prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _calculate_ema(self, window: int, prices: np.ndarray = None) -> np.ndarray:
        prices = np.array(prices) if prices is not None else np.array(self.prices)
        if len(prices) < window:
            return np.array([])
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        return np.convolve(prices, weights, mode='valid')

    def _calculate_macd(self) -> Dict:
        # Calculate EMAs as numpy arrays
        ema12 = self._calculate_ema(12)
        ema26 = self._calculate_ema(26)
        
        # Handle insufficient data
        if len(ema12) == 0 or len(ema26) == 0:
            return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}
        
        # Align lengths
        min_len = min(len(ema12), len(ema26))
        ema12 = ema12[-min_len:]
        ema26 = ema26[-min_len:]
        
        # Calculate MACD line
        macd_line = ema12 - ema26
        
        # Calculate signal line
        if len(macd_line) >= 9:
            signal_line = self._calculate_ema(9, macd_line)
            signal = signal_line[-1] if len(signal_line) > 0 else 0.0
        else:
            signal = 0.0
            
        return {
            "macd": float(macd_line[-1]),
            "signal": float(signal),
            "histogram": float(macd_line[-1] - signal)
        }

    async def analyze_market(self, prices: List[float], interval: str) -> Dict:
        # Check if the prices seem normalized (e.g. max price < 100)
        if max(prices) < 100:
            prices = [p * 1000 for p in prices]
        
        cache_key = hash(tuple(prices[-100:]))  # Cache based on price pattern
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        try:
            self.prices = prices
            analyzer = TechnicalAnalyzer(prices)
            
            # Calculate all indicators
            self.indicators = {
                "sma_20": self._calculate_sma(20),
                "sma_50": self._calculate_sma(50),
                "rsi": self._calculate_rsi(),
                "macd": self._calculate_macd(),
                "volatility": analyzer.calculate_volatility(),
                "key_levels": analyzer.find_key_levels()
            }

            messages = analyzer.generate_messages(self.indicators)
            
            gemini_analysis = await asyncio.to_thread(
                self.gemini.generate_analysis,
                {
                    "current_price": prices[-1],
                    "sma_20": self.indicators['sma_20'],
                    "sma_50": self.indicators['sma_50'],
                    "rsi": self.indicators['rsi'],
                    "macd": self.indicators['macd'],
                    "volatility": self.indicators['volatility'],
                    "key_levels": self.indicators['key_levels']
                }
            )

            result = {
                "metadata": {
                    "symbol": "BTCUSDT",  # This might need to be dynamic.
                    "interval": interval,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "data_quality": min(0.99, len(prices)/100)
                },
                "price_analysis": {
                    "current": prices[-1],
                    "prediction": self._generate_prediction(),
                    "confidence": self._calculate_confidence(),
                    **self.indicators
                },
                "frontend_insights": messages,
                "ai_insights": gemini_analysis
            }
            
            self.analysis_cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Analysis failed: " + str(e)
            )


    def _generate_prediction(self) -> float:
        """Combine indicators for price prediction"""
        base_price = self.indicators['sma_20']
        
        # MACD influence
        if self.indicators['macd']['histogram'] > 0:
            base_price *= 1.005
        else:
            base_price *= 0.995
            
        # RSI influence
        if self.indicators['rsi'] > 70:
            base_price *= 0.99
        elif self.indicators['rsi'] < 30:
            base_price *= 1.01
            
        return round(float(base_price), 2)

    def _calculate_confidence(self) -> float:
        """Calculate confidence score 0-1"""
        confidence = 0.5
        # Volatility impact
        confidence -= self.indicators['volatility'] * 0.5
        # SMA crossover confirmation
        if self.indicators['sma_20'] > self.indicators['sma_50']:
            confidence += 0.2
        # RSI confirmation
        if 30 < self.indicators['rsi'] < 70:
            confidence += 0.1
        return max(0.3, min(0.99, confidence))

predictor = AdvancedPredictor()