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
        """Generate market analysis using Gemini AI"""
        try:
            # Format numbers for better readability
            current_price = f"{data['current_price']:,.2f}"
            sma_20 = f"{data['sma_20']:,.2f}"
            sma_50 = f"{data['sma_50']:,.2f}"
            volatility = f"{data['volatility']*100:.2f}%"
            support = f"{data['key_levels']['support']:,.2f}"
            resistance = f"{data['key_levels']['resistance']:,.2f}"
            
            # Get trend direction
            trend = "bullish" if data['sma_20'] > data['sma_50'] else "bearish"
            
            prompt = f"""You are a professional cryptocurrency market analyst. Analyze these market indicators for trading insights:

Current Market Data:
- Price: ${current_price}
- 20 SMA: ${sma_20}
- 50 SMA: ${sma_50}
- RSI: {data['rsi']:.1f}
- Trend: {trend.upper()}
- Volatility: {volatility}
- Support: ${support}
- Resistance: ${resistance}

Provide a structured analysis in JSON format with:
1. "summary": One clear sentence about the current market state
2. "observations": List of 3 key technical observations
3. "recommendations": List of 2 specific trading suggestions
4. "risks": List of 2 potential risk factors

Focus on actionable insights and clear technical analysis. Keep it concise and professional.
"""

            # Generate with safety parameters
            response = self.model.generate_content(
                contents=[{
                    "parts": [{"text": prompt}]
                }],
                generation_config={
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40
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