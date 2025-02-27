# models.py
import numpy as np
from fastapi import HTTPException 
from datetime import datetime, timezone
from typing import List, Dict, Tuple
import logging
from services.gemini_insights import GeminiInsightsGenerator
from functools import lru_cache, wraps
import asyncio
from cachetools import TTLCache
from scipy.stats import linregress
from scipy.signal import savgol_filter

logger = logging.getLogger("CryptoPredictAPI")

class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=0.1**2):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = None
        self.estimate_error = 1.0

    def update(self, measurement):
        if self.estimate is None:
            self.estimate = measurement
            return measurement

        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance

        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error

        return self.estimate

class TechnicalAnalyzer:
    def __init__(self, prices: List[float], volumes: List[float] = None):
        self.prices = np.array(prices)
        self.volumes = np.array(volumes) if volumes is not None else None
        self.kf = KalmanFilter()
        self.filtered_prices = np.array([self.kf.update(p) for p in prices])
        
    def calculate_volatility(self, window: int = None) -> float:
        """Calculate adaptive volatility based on available data"""
        if window is None:
            window = min(24, len(self.prices) // 3)
        returns = np.diff(self.filtered_prices[-window:]) / self.filtered_prices[-window:-1]
        return float(np.std(returns))

    def find_key_levels(self) -> Dict:
        """Identify support/resistance using dynamic clustering"""
        lookback = min(100, len(self.prices))
        prices = self.filtered_prices[-lookback:]
        
        # Use smaller percentiles for limited data
        percentile_range = (30, 70) if len(prices) >= 50 else (40, 60)
        
        recent_highs = prices[prices >= np.percentile(prices, percentile_range[1])]
        recent_lows = prices[prices <= np.percentile(prices, percentile_range[0])]
        
        # Calculate trend for support/resistance adjustment
        trend = self._calculate_trend()
        trend_adjustment = trend * np.std(prices) * 0.1
        
        support = float(np.mean(recent_lows)) if len(recent_lows) > 0 else None
        resistance = float(np.mean(recent_highs)) if len(recent_highs) > 0 else None
        
        if support:
            support += trend_adjustment
        if resistance:
            resistance += trend_adjustment
            
        return {
            "support": support,
            "resistance": resistance,
            "trend_strength": abs(trend)
        }

    def _calculate_trend(self) -> float:
        """Calculate trend strength using linear regression"""
        x = np.arange(len(self.filtered_prices))
        slope, _, r_value, _, _ = linregress(x, self.filtered_prices)
        return slope * r_value**2  # Weighted by R-squared

    def calculate_volume_profile(self) -> Dict:
        """Analyze volume profile if available"""
        if self.volumes is None or len(self.volumes) < 2:
            return {"volume_trend": 0, "volume_signal": "neutral"}
            
        recent_vol = np.mean(self.volumes[-5:])
        older_vol = np.mean(self.volumes[-20:-5])
        vol_change = (recent_vol - older_vol) / older_vol
        
        return {
            "volume_trend": float(vol_change),
            "volume_signal": "increasing" if vol_change > 0.1 else "decreasing" if vol_change < -0.1 else "neutral"
        }

    def generate_messages(self, indicators: Dict) -> Dict:
        """Create human-readable insights with confidence levels"""
        messages = {
            "summary": "",
            "key_insights": [],
            "action_guide": {},
            "confidence_factors": {}
        }

        # Dynamic trend analysis
        trend_strength = abs(indicators.get('trend_strength', 0))
        if trend_strength > 0.5:
            confidence_mult = min(1.0, trend_strength)
            if indicators['sma_20'] > indicators['sma_50']:
                messages['key_insights'].append(f"Strong Bullish Trend (Confidence: {confidence_mult:.2f})")
            else:
                messages['key_insights'].append(f"Strong Bearish Trend (Confidence: {confidence_mult:.2f})")

        # RSI analysis with dynamic thresholds
        rsi_thresholds = (30, 70) if len(self.prices) >= 50 else (20, 80)
        if indicators['rsi'] > rsi_thresholds[1]:
            messages['key_insights'].append(f"Overbought (RSI: {indicators['rsi']:.1f})")
        elif indicators['rsi'] < rsi_thresholds[0]:
            messages['key_insights'].append(f"Oversold (RSI: {indicators['rsi']:.1f})")

        # Volume analysis if available
        vol_profile = self.calculate_volume_profile()
        if vol_profile['volume_signal'] != "neutral":
            messages['key_insights'].append(f"Volume {vol_profile['volume_signal']} ({vol_profile['volume_trend']:.1%} change)")

        # Generate weighted action guide
        bull_signals = sum(1 for insight in messages['key_insights'] if "Bullish" in insight or "Oversold" in insight)
        bear_signals = sum(1 for insight in messages['key_insights'] if "Bearish" in insight or "Overbought" in insight)
        
        # Weight signals by data quality
        data_quality = min(1.0, len(self.prices) / 100)
        signal_strength = (bull_signals - bear_signals) * data_quality
        
        messages['action_guide'] = {
            "buy": max(0.3, min(0.7, 0.5 + signal_strength * 0.1)),
            "sell": max(0.3, min(0.7, 0.5 - signal_strength * 0.1)),
            "hold": 0.5 + (0.1 if abs(signal_strength) < 0.2 else 0)
        }
        
        messages['confidence_factors'] = {
            "data_quality": data_quality,
            "trend_strength": trend_strength,
            "volume_confidence": 0.5 + abs(vol_profile['volume_trend'])
        }

        return messages

class AdvancedPredictor:
    def __init__(self):
        self.gemini = GeminiInsightsGenerator()
        self.analysis_cache = TTLCache(maxsize=500, ttl=300)
        self.prices = []
        self.volumes = []
        self.indicators = {
            "sma_20": None,
            "sma_50": None,
            "rsi": None,
            "macd": {},
            "trend": None,
            "volume_profile": None
        }
        
    def _calculate_sma(self, window: int) -> float:
        """Calculate SMA with dynamic window size for limited data"""
        if len(self.prices) < window:
            window = max(5, len(self.prices) // 2)
        return float(np.mean(self.prices[-window:]))

    def _calculate_adaptive_window(self, base_window: int) -> int:
        """Calculate adaptive window size based on available data"""
        return min(base_window, max(5, len(self.prices) // 3))

    def validate_data_length(window):
        def decorator(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                actual_window = self._calculate_adaptive_window(window)
                if len(self.prices) < actual_window:
                    raise ValueError(f"Need at least {actual_window} price points")
                return func(self, *args, actual_window=actual_window, **kwargs)
            return wrapper
        return decorator

    @validate_data_length(14)
    def _calculate_rsi(self, actual_window: int = 14) -> float:
        """Calculate RSI with noise filtering"""
        kf = KalmanFilter()
        filtered_prices = np.array([kf.update(p) for p in self.prices[-actual_window-1:]])
        deltas = np.diff(filtered_prices)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-6
        
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def _calculate_ema(self, window: int, prices: np.ndarray = None) -> np.ndarray:
        """Calculate EMA with Savitzky-Golay filtering for smoother results"""
        # Input validation
        if prices is None:
            prices = self.prices
        if not isinstance(prices, np.ndarray):
            prices = np.array(prices, dtype=float)
        if len(prices) == 0:
            return np.array([])
            
        # Ensure minimum window size
        if len(prices) < window:
            window = max(2, len(prices) // 2)
            
        # Apply Savitzky-Golay filter for noise reduction
        window_length = min(7, len(prices) - 1 if len(prices) % 2 == 0 else len(prices))
        if window_length > 2:
            try:
                prices = savgol_filter(prices, window_length, 3)
            except Exception as e:
                logger.warning(f"Savitzky-Golay filtering failed, using raw prices: {str(e)}")
            
        # Calculate EMA
        alpha = 2 / (window + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
            
        return ema

    @validate_data_length(26)
    def _calculate_macd(self, actual_window: int = 26) -> dict:
        """Calculate MACD with adaptive windows"""
        try:
            # Input validation
            if not isinstance(self.prices, (list, np.ndarray)) or len(self.prices) == 0:
                return {'macd_line': [0], 'signal_line': [0], 'histogram': [0]}

            prices = np.array(self.prices, dtype=float)
            
            # Ensure we have enough data
            min_required = actual_window + 10  # Add buffer for signal line
            if len(prices) < min_required:
                actual_window = max(10, len(prices) // 3)
            
            short_window = max(5, actual_window // 2)
            long_window = actual_window
            signal_window = max(3, actual_window // 3)
            
            # Calculate EMAs
            ema_short = self._calculate_ema(short_window, prices)
            ema_long = self._calculate_ema(long_window, prices)
            
            # Ensure we have valid data
            if len(ema_short) == 0 or len(ema_long) == 0:
                return {'macd_line': [0], 'signal_line': [0], 'histogram': [0]}
            
            # Align the arrays by trimming from the end
            min_len = min(len(ema_short), len(ema_long))
            macd_line = ema_short[-min_len:] - ema_long[-min_len:]
            
            # Calculate signal line
            signal_line = self._calculate_ema(signal_window, macd_line)
            
            # Ensure arrays are of equal length for histogram calculation
            min_len = min(len(macd_line), len(signal_line))
            if min_len == 0:
                return {'macd_line': [0], 'signal_line': [0], 'histogram': [0]}
                
            macd_line = macd_line[-min_len:]
            signal_line = signal_line[-min_len:]
            histogram = macd_line - signal_line
            
            return {
                'macd_line': macd_line.tolist(),
                'signal_line': signal_line.tolist(),
                'histogram': histogram.tolist()
            }
            
        except Exception as e:
            logger.error(f"MACD calculation error: {str(e)}")
            return {'macd_line': [0], 'signal_line': [0], 'histogram': [0]}

    def handle_analysis_errors(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except ValueError as e:
                logging.error(f"Validation error: {str(e)}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logging.error(f"Analysis error: {str(e)}")
                raise HTTPException(status_code=500, detail="Internal analysis error")
        return wrapper

    @handle_analysis_errors
    async def analyze_market(self, prices: List[float], volumes: List[float] = None, interval: str = "1h") -> Dict:
        # Input validation
        if not prices:
            raise ValueError("No price data provided")
            
        try:
            prices = [float(p) for p in prices]  # Ensure all prices are float
        except (TypeError, ValueError):
            raise ValueError("Invalid price data format")
            
        if len(prices) < 5:  # Minimum required data points
            raise ValueError("Need at least 5 price points for analysis")
            
        if volumes is not None:
            try:
                volumes = [float(v) for v in volumes]  # Ensure all volumes are float
            except (TypeError, ValueError):
                volumes = None  # Reset to None if invalid
                
        # Remove price normalization - work with actual prices
        self.prices = np.array(prices, dtype=float)
        self.volumes = np.array(volumes, dtype=float) if volumes is not None else None
        
        cache_key = hash(tuple(prices[-100:]))
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        try:
            analyzer = TechnicalAnalyzer(self.prices, self.volumes)
            
            # Calculate indicators with noise reduction and error handling
            self.indicators = {}
            
            try:
                self.indicators["sma_20"] = self._calculate_sma(20)
            except Exception as e:
                logger.error(f"SMA-20 calculation failed: {str(e)}")
                self.indicators["sma_20"] = float(prices[-1])
                
            try:
                self.indicators["sma_50"] = self._calculate_sma(50)
            except Exception as e:
                logger.error(f"SMA-50 calculation failed: {str(e)}")
                self.indicators["sma_50"] = float(prices[-1])
                
            try:
                self.indicators["rsi"] = self._calculate_rsi()
            except Exception as e:
                logger.error(f"RSI calculation failed: {str(e)}")
                self.indicators["rsi"] = 50.0
                
            try:
                self.indicators["macd"] = self._calculate_macd()
            except Exception as e:
                logger.error(f"MACD calculation failed: {str(e)}")
                self.indicators["macd"] = {'macd_line': [0], 'signal_line': [0], 'histogram': [0]}
                
            try:
                self.indicators["volatility"] = float(analyzer.calculate_volatility())
            except Exception as e:
                logger.error(f"Volatility calculation failed: {str(e)}")
                self.indicators["volatility"] = 0.01
                
            try:
                self.indicators["key_levels"] = analyzer.find_key_levels()
            except Exception as e:
                logger.error(f"Key levels calculation failed: {str(e)}")
                current_price = float(prices[-1])
                self.indicators["key_levels"] = {
                    "support": current_price * 0.95,
                    "resistance": current_price * 1.05,
                    "trend_strength": 0.0
                }
                
            try:
                self.indicators["volume_profile"] = analyzer.calculate_volume_profile()
            except Exception as e:
                logger.error(f"Volume profile calculation failed: {str(e)}")
                self.indicators["volume_profile"] = {"volume_trend": 0.0, "volume_signal": "neutral"}

            try:
                messages = analyzer.generate_messages(self.indicators)
            except Exception as e:
                logger.error(f"Message generation failed: {str(e)}")
                messages = {
                    "summary": "Analysis available but insights generation failed",
                    "key_insights": [],
                    "action_guide": {"buy": 0.5, "sell": 0.5, "hold": 0.5},
                    "confidence_factors": {"data_quality": 0.5, "trend_strength": 0, "volume_confidence": 0.5}
                }
            
            # Get AI insights with confidence weighting
            try:
                gemini_analysis = await asyncio.to_thread(
                    self.gemini.generate_analysis,
                    {
                        "current_price": float(prices[-1]),
                        **self.indicators,
                        "data_quality": min(0.99, len(prices)/100),
                        "interval": interval
                    }
                )
            except Exception as e:
                logger.error(f"Gemini analysis failed: {str(e)}")
                gemini_analysis = {"error": "AI analysis temporarily unavailable"}

            result = {
                "metadata": {
                    "interval": interval,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "data_quality": min(0.99, len(prices)/100),
                    "confidence_score": self._calculate_confidence()
                },
                "price_analysis": {
                    "current": float(prices[-1]),
                    "prediction": self._generate_prediction(),
                    "prediction_range": self._calculate_prediction_range(),
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
                detail=f"Analysis failed: {str(e)}"
            )

    def _generate_prediction(self) -> float:
        """Generate price prediction using multiple factors"""
        try:
            # Start with current price as base
            current_price = self.prices[-1]
            
            # Calculate percentage changes for prediction
            trend = self.indicators['key_levels']['trend_strength']
            trend_impact = np.tanh(trend * 0.1)  # Normalized trend impact
            
            # Apply MACD momentum as percentage
            macd_hist = self.indicators['macd']['histogram'][-1]
            macd_impact = np.tanh(macd_hist * 0.1)  # Normalized impact
            
            # Apply RSI mean reversion as percentage
            rsi_impact = np.tanh((50 - self.indicators['rsi']) * 0.02)  # Normalized RSI impact
            
            # Volume impact if available
            volume_impact = 0
            if self.indicators['volume_profile']['volume_signal'] != "neutral":
                volume_impact = np.tanh(0.1 * self.indicators['volume_profile']['volume_trend'])
            
            # Combine factors with data quality weighting
            data_quality = min(1.0, len(self.prices) / 100)
            total_impact = (trend_impact + macd_impact + rsi_impact + volume_impact) * data_quality
            
            # Apply impact as percentage of current price (max 5% change)
            prediction = current_price * (1 + np.clip(total_impact * 0.05, -0.05, 0.05))
            
            return round(float(prediction), 8 if current_price < 1 else 6 if current_price < 10 else 2)
        except Exception as e:
            logger.error(f"Prediction calculation error: {str(e)}")
            return self.prices[-1]  # Return current price as fallback

    def _calculate_prediction_range(self) -> Dict[str, float]:
        """Calculate prediction range based on volatility and confidence"""
        try:
            volatility = self.indicators['volatility']
            confidence = self._calculate_confidence()
            base_prediction = self._generate_prediction()
            current_price = self.prices[-1]
            
            # Wider range for lower confidence (max 10% range)
            range_multiplier = np.clip(2 * (1 + (1 - confidence)) * volatility, 0.001, 0.1)
            
            # Calculate range as percentage of base prediction
            range_low = base_prediction * (1 - range_multiplier)
            range_high = base_prediction * (1 + range_multiplier)
            
            # Ensure correct ordering
            if range_low > range_high:
                range_low, range_high = range_high, range_low
                
            # Use appropriate decimal places based on price scale
            decimals = 8 if current_price < 1 else 6 if current_price < 10 else 2
            return {
                "low": round(float(range_low), decimals),
                "high": round(float(range_high), decimals)
            }
        except Exception as e:
            logger.error(f"Range calculation error: {str(e)}")
            current_price = self.prices[-1]
            # Fallback to ±0.5% of current price
            decimals = 8 if current_price < 1 else 6 if current_price < 10 else 2
            return {
                "low": round(float(current_price * 0.995), decimals),
                "high": round(float(current_price * 1.005), decimals)
            }

    def _calculate_confidence(self) -> float:
        """Calculate enhanced confidence score 0-1"""
        if not self.indicators:
            return 0.3
            
        confidence = 0.5
        
        # Data quality impact
        data_quality = min(1.0, len(self.prices) / 100)
        confidence *= data_quality
        
        # Trend strength impact
        trend_strength = self.indicators['key_levels']['trend_strength']
        confidence += 0.1 * min(1.0, abs(trend_strength))
        
        # Volatility impact (inverse)
        confidence -= self.indicators['volatility'] * 0.5
        
        # Volume confirmation if available
        if self.indicators['volume_profile']['volume_signal'] != "neutral":
            volume_trend = abs(self.indicators['volume_profile']['volume_trend'])
            confidence += 0.1 * min(1.0, volume_trend)
        
        # Technical indicator agreement
        if 30 < self.indicators['rsi'] < 70:
            confidence += 0.1
            
        macd_hist = self.indicators['macd']['histogram'][-1]
        if abs(macd_hist) > 0:  # Strong MACD signal
            confidence += 0.1
            
        return max(0.3, min(0.95, confidence))

predictor = AdvancedPredictor()