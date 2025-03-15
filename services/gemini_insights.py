import os
import json
import logging
import google.generativeai as genai
from typing import Dict, Optional

# Configure logging first to prevent STDERR warnings
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("GeminiAI")

class GeminiInsightsGenerator:
    def __init__(self):
        """Initialize Gemini AI with environment configuration"""
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            logger.error("Missing GEMINI_API_KEY in environment variables")
            raise ValueError(
                "API key required. Set GEMINI_API_KEY in .env file or environment variables"
            )
            
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
            
            # Test the API connection
            response = self.model.generate_content("Test connection")
            if not response or not response.text:
                raise ConnectionError("Failed to connect to Gemini API")
                
        except Exception as e:
            logger.error(f"Gemini initialization failed: {str(e)}")
            raise

    def generate_analysis(self, data: Dict) -> Dict:
        """Generate market analysis using Gemini AI with asset-specific insights"""
        try:
            # Determine price scale and format numbers accordingly
            current_price = data['current_price']
            price_format = "{:,.8f}" if current_price < 0.01 else "{:,.6f}" if current_price < 1 else "{:,.2f}"
            
            # Format numbers with appropriate decimal places
            formatted_data = {
                "price": price_format.format(current_price),
                "sma_20": price_format.format(data['sma_20']),
                "sma_50": price_format.format(data['sma_50']),
                "support": price_format.format(data['key_levels']['support']),
                "resistance": price_format.format(data['key_levels']['resistance'])
            }
            
            # Get trend direction and strength
            trend = "bullish" if data['sma_20'] > data['sma_50'] else "bearish"
            trend_strength = abs(data['key_levels']['trend_strength'])
            trend_desc = "strong" if trend_strength > 0.5 else "moderate" if trend_strength > 0.2 else "weak"
            
            # Calculate percentage changes for better context
            sma20_change = ((data['sma_20'] - current_price) / current_price) * 100
            sma50_change = ((data['sma_50'] - current_price) / current_price) * 100

            # Determine if this is a major pair
            is_major_pair = (
                current_price > 100  # High unit price
                or "BTC" in data.get('symbol', '')  # Bitcoin pairs
                or "ETH" in data.get('symbol', '')  # Ethereum pairs
                or "USDT" in data.get('symbol', '')  # Major stablecoin pairs
            )
            
            if is_major_pair:
                prompt = f"""You are a professional cryptocurrency market analyst focusing on major cryptocurrency pairs. Analyze these market indicators for trading insights:

Current Market Data for {data.get('symbol', 'Major Pair')}:
- Current Price: ${formatted_data['price']}
- 20 SMA: ${formatted_data['sma_20']} ({sma20_change:+.2f}% from current)
- 50 SMA: ${formatted_data['sma_50']} ({sma50_change:+.2f}% from current)
- RSI: {data['rsi']:.1f}
- Trend: {trend_desc.upper()} {trend.upper()}
- Volatility: {data['volatility']*100:.2f}%
- Support: ${formatted_data['support']}
- Resistance: ${formatted_data['resistance']}

Provide a structured analysis in JSON format with:
1. "summary": One clear sentence about the current market state
2. "observations": List of 3 key technical observations
3. "recommendations": List of 2 specific trading suggestions with clear entry/exit points
4. "risks": List of 2 potential risk factors to monitor

Focus on actionable insights and clear technical analysis. Keep it concise and professional."""
            else:
                prompt = f"""You are a professional cryptocurrency market analyst specializing in small-cap and micro-cap assets. Analyze these market indicators for trading insights:

Current Market Data for Small-Cap Asset {data.get('symbol', '')}:
- Current Price: ${formatted_data['price']}
- 20 SMA: ${formatted_data['sma_20']} ({sma20_change:+.2f}% from current)
- 50 SMA: ${formatted_data['sma_50']} ({sma50_change:+.2f}% from current)
- RSI: {data['rsi']:.1f}
- Trend: {trend_desc.upper()} {trend.upper()}
- Volatility: {data['volatility']*100:.2f}%
- Support: ${formatted_data['support']}
- Resistance: ${formatted_data['resistance']}

Consider:
1. This is a small-cap asset with potentially limited trading history
2. Price movements may be more volatile than major pairs
3. Lower liquidity may affect price action
4. Technical indicators may be less reliable

Provide a structured analysis in JSON format with:
1. "summary": One clear sentence about the current market state, considering the asset's small-cap nature
2. "observations": List of 3 key technical observations, focusing on reliable signals
3. "recommendations": List of 2 specific trading suggestions, emphasizing risk management for small-cap assets
4. "risks": List of 2 potential risk factors specific to small-cap trading

Focus on practical insights while acknowledging the limitations of technical analysis for small-cap assets."""

            # Generate with adjusted parameters based on asset type
            response = self.model.generate_content(
                contents=[{
                    "parts": [{"text": prompt}]
                }],
                generation_config={
                    "temperature": 0.7 if is_major_pair else 0.6,  # More conservative for small caps
                    "top_p": 0.8 if is_major_pair else 0.7,       # More focused for small caps
                    "top_k": 40 if is_major_pair else 30          # More precise for small caps
                },
                request_options={"timeout": 15}
            )
            
            if not response or not response.text:
                return {"error": "Empty response from AI"}
                
            return self._parse_response(response.text)
            
        except genai.GenerationError as e:
            logger.error(f"Generation error: {str(e)}")
            return {
                "error": "AI analysis service unavailable",
                "details": str(e)
            }
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "error": "Failed to generate analysis",
                "details": str(e)
            }

    def _parse_response(self, text: str) -> Dict:
        """Parse and sanitize Gemini response"""
        try:
            # Clean up the response text
            clean_text = (text.strip()
                         .replace("```json", "")
                         .replace("```", "")
                         .replace("\n", " ")
                         .strip())
            
            # Handle empty responses
            if not clean_text:
                return {"error": "Empty AI response"}
            
            try:
                parsed = json.loads(clean_text)
            except json.JSONDecodeError:
                # Try to extract JSON from the text if it's wrapped in other content
                import re
                json_match = re.search(r'\{.*\}', clean_text)
                if json_match:
                    parsed = json.loads(json_match.group(0))
                else:
                    raise
            
            # Validate response structure
            required_keys = ["summary", "observations", "recommendations", "risks"]
            if not all(key in parsed for key in required_keys):
                return {
                    "error": "Invalid analysis format",
                    "response": clean_text[:200]
                }
            
            # Ensure all lists have the correct number of items
            if not (len(parsed["observations"]) == 3 and 
                   len(parsed["recommendations"]) == 2 and 
                   len(parsed["risks"]) == 2):
                logger.warning("Response lists have incorrect lengths")
            
            return {
                "market_summary": parsed["summary"],
                "technical_observations": parsed["observations"],
                "trading_recommendations": parsed["recommendations"],
                "risk_factors": parsed["risks"]
            }
            
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response")
            return {
                "error": "Invalid response format",
                "raw_response": clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
            }
        except Exception as e:
            logger.error(f"Parsing failed: {str(e)}")
            return {"error": "Unexpected parsing error"}