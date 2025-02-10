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
            # Simplified configuration without unsupported options
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        except Exception as e:
            logger.error(f"Gemini initialization failed: {str(e)}")
            raise

    def generate_analysis(self, data: Dict) -> Dict:
        """Generate market analysis using Gemini AI"""
        try:
            prompt = f"""Analyze these cryptocurrency market indicators:
            
            Current Price: {data['current_price']}
            20-period SMA: {data['sma_20']}
            50-period SMA: {data['sma_50']}
            RSI: {data['rsi']}
            MACD Histogram: {data['macd']['histogram']}
            Volatility: {round(data['volatility']*100, 2)}%
            Support Level: {data['key_levels']['support']}
            Resistance Level: {data['key_levels']['resistance']}

            Provide structured response with:
            1. One-sentence market summary
            2. Three technical observations
            3. Two trading recommendations
            4. Two risk factors

            Use JSON format with keys: summary, observations, recommendations, risks
            """

            # Add timeout directly in the generation call
            response = self.model.generate_content(
                prompt,
                request_options={"timeout": 10}  # 10-second timeout here
            )
            
            return self._parse_response(response.text)
            
        except genai.GenerationError as e:
            logger.error(f"Generation error: {str(e)}")
            return {"error": "AI analysis service unavailable"}
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {"error": "Failed to generate analysis"}

    # Keep the rest of the _parse_response method unchanged

    def _parse_response(self, text: str) -> Dict:
        """Parse and sanitize Gemini response"""
        try:
            # Clean unexpected formatting
            clean_text = text.strip().replace("```json", "").replace("```", "")
            
            # Handle empty responses
            if not clean_text:
                return {"error": "Empty AI response"}
                
            parsed = json.loads(clean_text)
            
            # Validate response structure
            required_keys = ["summary", "observations", "recommendations", "risks"]
            if not all(key in parsed for key in required_keys):
                return {"error": "Invalid analysis format", "response": clean_text[:200]}
                
            return parsed
            
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response")
            return {
                "error": "Invalid response format",
                "raw_response": clean_text[:200] + "..." if len(clean_text) > 200 else clean_text
            }
        except Exception as e:
            logger.error(f"Parsing failed: {str(e)}")
            return {"error": "Unexpected parsing error"}