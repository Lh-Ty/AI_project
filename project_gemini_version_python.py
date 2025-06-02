# app.py
import os
import asyncio
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

def init():
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'
    os.environ["GOOGLE_API_KEY"]="AIzaSyDpPurzES8YtReAZJEsXcld2Nnfgc6mT94" # 替换为你的实际API密钥
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = "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" # 替换为你的实际API密钥
    os.environ["LANGSMITH_PROJECT"] = "AI Couple Compatibility Analyzer"
init()
# --- Load Environment Variables ---
load_dotenv()

# --- LangSmith Configuration ---
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper function for Zodiac Sign ---
def get_zodiac_sign(month: int, day: int) -> str:
    if (month == 1 and day >= 20) or (month == 2 and day <= 18):
        return "水瓶座 (Aquarius)"
    elif (month == 2 and day >= 19) or (month == 3 and day <= 20):
        return "双鱼座 (Pisces)"
    elif (month == 3 and day >= 21) or (month == 4 and day <= 19):
        return "白羊座 (Aries)"
    elif (month == 4 and day >= 20) or (month == 5 and day <= 20):
        return "金牛座 (Taurus)"
    elif (month == 5 and day >= 21) or (month == 6 and day <= 21):
        return "双子座 (Gemini)"
    elif (month == 6 and day >= 22) or (month == 7 and day <= 22):
        return "巨蟹座 (Cancer)"
    elif (month == 7 and day >= 23) or (month == 8 and day <= 22):
        return "狮子座 (Leo)"
    elif (month == 8 and day >= 23) or (month == 9 and day <= 22):
        return "处女座 (Virgo)"
    elif (month == 9 and day >= 23) or (month == 10 and day <= 23):
        return "天秤座 (Libra)"
    elif (month == 10 and day >= 24) or (month == 11 and day <= 22):
        return "天蝎座 (Scorpio)"
    elif (month == 11 and day >= 23) or (month == 12 and day <= 21):
        return "射手座 (Sagittarius)"
    elif (month == 12 and day >= 22) or (month == 1 and day <= 19):
        return "摩羯座 (Capricorn)"
    return "未知星座"


from typing import Optional, List

# --- Pydantic Models for Structured Output ---
class ImageAnalysis(BaseModel):
    guessed_gender: str = Field(description="Gender guess based on visual cues (e.g., male, female, undeterminable)")
    visual_description: str = Field(description="Brief description of appearance and demeanor")
    fictional_bazi_traits: List[str] = Field(description="Fictional elemental traits based on visual appearance (for entertainment only)")
    provided_gender: Optional[str] = Field(default=None, description="Gender provided by the user.")
    provided_birthday: Optional[str] = Field(default=None, description="Birthday provided by the user (YYYY-MM-DD format).")
    zodiac_sign: Optional[str] = Field(default=None, description="Zodiac sign derived from provided birthday.") # 新增星座字段
    hobbies: Optional[List[str]] = Field(default=None, description="Hobbies provided by the user.")
    personality: Optional[str] = Field(default=None, description="Personality traits provided by the user.")



class CompatibilityResult(BaseModel):
    """Represents the couple compatibility result."""
    compatibility_score: int = Field(description="Compatibility score (0-100)")
    compatibility_explanation: str = Field(description="Explanation of the compatibility score (for entertainment only)")

# --- Prompts ---
IMAGE_ANALYSIS_PROMPT_TEXT = """
You are a neutral, factual, and precise AI assistant. Please analyze the person in this image.
Based solely on the person’s visual appearance and demeanor in a single image, provide a valid JSON object containing the following keys:

"guessed_gender": Your best guess of the person’s gender based on visual cues only (e.g., "male", "female", "undeterminable").
"visual_description": A concise, neutral, and respectful description of the person’s appearance and any noticeable emotions or personality traits (in Chinese).
"fictional_bazi_traits": For entertainment purposes only, creatively and metaphorically invent 2–3 “elemental” or “personality” traits inspired by the ancient Chinese Five Elements (Wood, Fire, Earth, Metal, Water). Base these purely on the person’s visual impression or perceived aura (in Chinese). These fictional traits should be purely imaginative and not based on real birth data or fortune-telling.
Output a valid JSON object in the following format:{
  "guessed_gender": "string",
  "visual_description": "string",
  "fictional_bazi_traits": ["string", "string", "string"]
}
"""

COMPATIBILITY_PROMPT_TEXT = """
You are a humorous and creative AI assistant. You received analysis results for two people.
Here's the detailed information for Person 1:
- Visual Analysis: {analysis1_visual_json_str}
- Provided Information:
  - Gender: {provided_gender1}
  - Birthday: {provided_birthday1}
  - Zodiac Sign: {zodiac_sign1}
  - Hobbies: {provided_hobbies1}
  - Personality: {provided_personality1}

Here's the detailed information for Person 2:
- Visual Analysis: {analysis2_visual_json_str}
- Provided Information:
  - Gender: {provided_gender2}
  - Birthday: {provided_birthday2}
  - Zodiac Sign: {zodiac_sign2}
  - Hobbies: {provided_hobbies2}
  - Personality: {provided_personality2}

Considering all the available information (AI's visual analysis, user-provided gender, birthday, zodiac sign, hobbies, and personality), generate:
1. A "compatibility score" between 0-100.
2. A short, positive, fun explanation focusing on how their combined traits and interests might complement each other, with a creative and entertaining tone.
Emphasize that this is for creative entertainment only and not real relationship advice.

Output JSON with these keys:
{{
  "compatibility_score": integer,
  "compatibility_explanation": "string"
}}
"""

# --- LangChain LLMs and Parsers ---
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("CRITICAL: GOOGLE_API_KEY environment variable not set.")
    
multimodal_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    generation_config={"response_mime_type": "application/json"}
)

text_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    generation_config={"response_mime_type": "application/json"}
)

image_analysis_parser = JsonOutputParser(pydantic_object=ImageAnalysis)
compatibility_parser = JsonOutputParser(pydantic_object=CompatibilityResult)

# --- Core Logic Functions ---
async def analyze_single_image_async(image_base64: str, person_identifier: str) -> dict:
    logger.info(f"Analyzing image for person {person_identifier}...")
    message_content = [
        {"type": "text", "text": IMAGE_ANALYSIS_PROMPT_TEXT},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        }
    ]
    human_message = HumanMessage(content=message_content)
    chain = multimodal_llm | JsonOutputParser() # Use generic JSON parser here first

    try:
        result_raw = await chain.ainvoke([human_message])
        logger.info(f"Completed visual analysis for person {person_identifier}.")
        # Validate and return a dictionary that matches the expected Pydantic model structure
        return result_raw
    except Exception as e:
        logger.error(f"Image analysis error for person {person_identifier}: {e}", exc_info=True)
        raise

async def calculate_compatibility_async(
    analysis1_visual_data: dict, provided_data1: dict, full_analysis1: ImageAnalysis,
    analysis2_visual_data: dict, provided_data2: dict, full_analysis2: ImageAnalysis
) -> CompatibilityResult:
    logger.info("Calculating couple compatibility...")
    prompt_template = ChatPromptTemplate.from_template(COMPATIBILITY_PROMPT_TEXT)
    chain = prompt_template | text_llm | compatibility_parser

    # Convert dictionaries to JSON strings for the prompt
    analysis1_visual_json_str = jsonify(analysis1_visual_data).get_data(as_text=True)
    analysis2_visual_json_str = jsonify(analysis2_visual_data).get_data(as_text=True)

    try:
        result = await chain.ainvoke({
            "analysis1_visual_json_str": analysis1_visual_json_str,
            "provided_gender1": provided_data1.get('gender', '未提供'),
            "provided_birthday1": provided_data1.get('birthday', '未提供'),
            "zodiac_sign1": full_analysis1.zodiac_sign or '未提供', # 使用完整的分析对象中的星座
            "provided_hobbies1": provided_data1.get('hobbies', '未提供'),
            "provided_personality1": provided_data1.get('personality', '未提供'),
            
            "analysis2_visual_json_str": analysis2_visual_json_str,
            "provided_gender2": provided_data2.get('gender', '未提供'),
            "provided_birthday2": provided_data2.get('birthday', '未提供'),
            "zodiac_sign2": full_analysis2.zodiac_sign or '未提供', # 使用完整的分析对象中的星座
            "provided_hobbies2": provided_data2.get('hobbies', '未提供'),
            "provided_personality2": provided_data2.get('personality', '未提供'),
        })
        logger.info("Compatibility calculation completed.")
        return result
    except Exception as e:
        logger.error(f"Compatibility calculation error: {e}", exc_info=True)
        raise

# --- API Endpoint ---
@app.route('/api/analyze_couple', methods=['POST'])
def analyze_couple_endpoint():
    if not request.is_json:
        logger.warning("Request was not JSON.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    image1_base64 = data.get('image1_base64')
    image2_base64 = data.get('image2_base64')

    # Extract user-provided data
    provided_data1 = {
        'gender': data.get('gender1'),
        'birthday': data.get('birthday1'),
        'hobbies': data.get('hobbies1').split(',') if data.get('hobbies1') else [],
        'personality': data.get('personality1')
    }
    provided_data2 = {
        'gender': data.get('gender2'),
        'birthday': data.get('birthday2'),
        'hobbies': data.get('hobbies2').split(',') if data.get('hobbies2') else [],
        'personality': data.get('personality2')
    }


    if not image1_base64 or not image2_base64:
        logger.warning("Missing image data in request.")
        return jsonify({"error": "Missing image1_base64 or image2_base64"}), 400

    logger.info("Received /api/analyze_couple request...")

    try:
        async def _run_analysis_pipeline():
            # Perform visual analysis concurrently
            analysis1_visual_task = analyze_single_image_async(image1_base64, "1")
            analysis2_visual_task = analyze_single_image_async(image2_base64, "2")
            
            person1_visual_analysis, person2_visual_analysis = await asyncio.gather(
                analysis1_visual_task,
                analysis2_visual_task
            )

            # Determine zodiac sign from provided birthday
            zodiac_sign1 = None
            if provided_data1['birthday']:
                try:
                    bday1 = datetime.strptime(provided_data1['birthday'], '%Y-%m-%d')
                    zodiac_sign1 = get_zodiac_sign(bday1.month, bday1.day)
                except ValueError:
                    logger.warning(f"Invalid birthday format for person 1: {provided_data1['birthday']}")
            
            zodiac_sign2 = None
            if provided_data2['birthday']:
                try:
                    bday2 = datetime.strptime(provided_data2['birthday'], '%Y-%m-%d')
                    zodiac_sign2 = get_zodiac_sign(bday2.month, bday2.day)
                except ValueError:
                    logger.warning(f"Invalid birthday format for person 2: {provided_data2['birthday']}")

            # Combine visual analysis with user-provided data for each person
            # Create Pydantic objects for validation and consistent structure
            person1_full_analysis = ImageAnalysis(
                guessed_gender=person1_visual_analysis.get('guessed_gender', 'undeterminable'),
                visual_description=person1_visual_analysis.get('visual_description', ''),
                fictional_bazi_traits=person1_visual_analysis.get('fictional_bazi_traits', []),
                provided_gender=provided_data1['gender'],
                provided_birthday=provided_data1['birthday'],
                zodiac_sign=zodiac_sign1, # 添加计算出的星座
                hobbies=provided_data1['hobbies'],
                personality=provided_data1['personality']
            )
            person2_full_analysis = ImageAnalysis(
                guessed_gender=person2_visual_analysis.get('guessed_gender', 'undeterminable'),
                visual_description=person2_visual_analysis.get('visual_description', ''),
                fictional_bazi_traits=person2_visual_analysis.get('fictional_bazi_traits', []),
                provided_gender=provided_data2['gender'],
                provided_birthday=provided_data2['birthday'],
                zodiac_sign=zodiac_sign2, # 添加计算出的星座
                hobbies=provided_data2['hobbies'],
                personality=provided_data2['personality']
            )
            
            compatibility = await calculate_compatibility_async(
                person1_visual_analysis, provided_data1, person1_full_analysis, # 传递完整的分析对象
                person2_visual_analysis, provided_data2, person2_full_analysis  # 传递完整的分析对象
            )
            
            return {
                "person1_analysis": person1_full_analysis.dict(),
                "person2_analysis": person2_full_analysis.dict(),
                "compatibility_result": compatibility
            }

        result_data = asyncio.run(_run_analysis_pipeline())
        logger.info("/api/analyze_couple request processed successfully.")
        return jsonify(result_data)

    except Exception as e:
        logger.error(f"/api/analyze_couple processing error: {e}", exc_info=True)
        return jsonify({"error": "Internal analysis error, please try again later."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    init()
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
    else:
        print("AI Couple Compatibility Analyzer backend starting...")
        print("Ensure your GOOGLE_API_KEY and LangSmith (optional) are configured.")
        print("Service will run at http://localhost:5001")
        app.run(debug=True, port=5001, host='0.0.0.0')