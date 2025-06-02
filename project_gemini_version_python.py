# app.py
import os
import asyncio
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain_community.chat_models.tongyi import ChatTongyi

def init():
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'
    os.environ["GOOGLE_API_KEY"]="AIzaSyDpPurzES8YtReAZJEsXcld2Nnfgc6mT94" # Replace with your actual API key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = "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" # Replace with your actual API key
    os.environ["LANGSMITH_PROJECT"] = "AI Couple Compatibility Analyzer"
    os.environ["DASHSCOPE_API_KEY"] = 'sk-c7bcac6af25d4fe8ad9907a1049b1363'
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
        return "Aquarius"
    elif (month == 2 and day >= 19) or (month == 3 and day <= 20):
        return "Pisces"
    elif (month == 3 and day >= 21) or (month == 4 and day <= 19):
        return "Aries"
    elif (month == 4 and day >= 20) or (month == 5 and day <= 20):
        return "Taurus"
    elif (month == 5 and day >= 21) or (month == 6 and day <= 21):
        return "Gemini"
    elif (month == 6 and day >= 22) or (month == 7 and day <= 22):
        return "Cancer"
    elif (month == 7 and day >= 23) or (month == 8 and day <= 22):
        return "Leo"
    elif (month == 8 and day >= 23) or (month == 9 and day <= 22):
        return "Virgo"
    elif (month == 9 and day >= 23) or (month == 10 and day <= 23):
        return "Libra"
    elif (month == 10 and day >= 24) or (month == 11 and day <= 22):
        return "Scorpio"
    elif (month == 11 and day >= 23) or (month == 12 and day <= 21):
        return "Sagittarius"
    elif (month == 12 and day >= 22) or (month == 1 and day <= 19):
        return "Capricorn"
    return "Unknown Zodiac Sign"

# --- Pydantic Models for Structured Output ---
class ImageAnalysis(BaseModel):
    guessed_gender: str = Field(description="Gender guess based on visual cues (e.g., male, female, undeterminable)")
    visual_description: str = Field(description="Brief description of appearance and demeanor")
    fictional_bazi_traits: List[str] = Field(description="Fictional elemental traits based on visual appearance (for entertainment only)")
    provided_gender: Optional[str] = Field(default=None, description="Gender provided by the user.")
    provided_birthday: Optional[str] = Field(default=None, description="Birthday provided by the user (YYYY-MM-DD format).")
    zodiac_sign: Optional[str] = Field(default=None, description="Zodiac sign derived from provided birthday.")
    hobbies: Optional[List[str]] = Field(default=None, description="Hobbies provided by the user.")
    personality: Optional[str] = Field(default=None, description="Personality traits provided by the user.")

class CompatibilityResult(BaseModel):
    """Represents the couple compatibility result."""
    compatibility_score: int = Field(description="Compatibility score (0-100)")
    compatibility_explanation: str = Field(description="Explanation of the compatibility score, considering visual traits, provided information, and simulated dialogue (for entertainment only)")
    dialogue_compatibility_assessment: Optional[str] = Field(default=None, description="Assessment of compatibility based on the simulated dialogue.")

class DialogueTurn(BaseModel):
    speaker_id: str = Field(description="Speaker ID ('person1' or 'person2')")
    speaker_name: str = Field(description="Speaker name (e.g., 'Person 1' or 'Person 2')")
    utterance: str = Field(description="Content of the utterance")

class TopicDialogue(BaseModel):
    topic: str = Field(description="Dialogue topic")
    dialogue_history: List[DialogueTurn] = Field(description="Dialogue history for this topic")

class CoupleDialogueResult(BaseModel):
    dialogues_by_topic: List[TopicDialogue] = Field(description="Complete dialogue results organized by topic")
    person1_name_used: str = Field(default="Person 1")
    person2_name_used: str = Field(default="Person 2")

# --- Prompts ---
IMAGE_ANALYSIS_PROMPT_TEXT = """
You are a neutral, factual, and precise AI assistant. Please analyze the person in this image.
Based solely on the person’s visual appearance and demeanor in a single image, provide a valid JSON object containing the following keys:

"guessed_gender": Your best guess of the person’s gender based on visual cues only (e.g., "male", "female", "undeterminable").
"visual_description": A concise, neutral, and respectful description of the person’s appearance and any noticeable emotions or personality traits (in English).
"fictional_bazi_traits": For entertainment purposes only, creatively and metaphorically invent 2–3 “elemental” or “personality” traits inspired by the ancient Chinese Five Elements (Wood, Fire, Earth, Metal, Water). Base these purely on the person’s visual impression or perceived aura (in English). These fictional traits should be purely imaginative and not based on real birth data or fortune-telling.
Output a valid JSON object in the following format:{
  "guessed_gender": "string",
  "visual_description": "string",
  "fictional_bazi_traits": ["string", "string", "string"]
}
"""

COMPATIBILITY_PROMPT_TEXT = """
You are a humorous and creative AI assistant. You will receive analysis results for two people, along with a simulated dialogue between them.
Please provide a couple compatibility report based on all the following information:

Person 1's detailed information:
- Visual Analysis: {analysis1_visual_json_str}
- Provided Information:
  - Gender: {provided_gender1}
  - Birthday: {provided_birthday1}
  - Zodiac Sign: {zodiac_sign1}
  - Hobbies: {provided_hobbies1}
  - Personality: {provided_personality1}

Person 2's detailed information:
- Visual Analysis: {analysis2_visual_json_str}
- Provided Information:
  - Gender: {provided_gender2}
  - Birthday: {provided_birthday2}
  - Zodiac Sign: {zodiac_sign2}
  - Hobbies: {provided_hobbies2}
  - Personality: {provided_personality2}

Simulated Dialogue Content:
{dialogue_summary}

Based on all the above information (AI's visual analysis, user-provided information, zodiac signs, hobbies, personality, and simulated dialogue content), generate:
1. A "compatibility score" between 0-100.
2. A short, positive, and fun explanation (compatibility_explanation), focusing on how their combined traits and interests might complement each other, and reflecting the rapport shown in the dialogue.
3. An assessment based on the simulated dialogue content (dialogue_compatibility_assessment), explaining how the dialogue reflects their compatibility, such as their interaction style, discovery of common interests, etc.

Please emphasize that all results are for creative entertainment only and do not constitute real relationship advice.

Output JSON with these keys:
{{
  "compatibility_score": integer,
  "compatibility_explanation": "string",
  "dialogue_compatibility_assessment": "string"
}}
"""

COUPLE_DIALOGUE_SYSTEM_PROMPT = """You will play one of two roles and engage in a light-hearted, romantic conversation centered around a specified topic.
Please respond based on your character's profile and the provided dialogue history. Your goal is to make the conversation natural, interesting, and to express the character's personality and mutual affection.
Only output your current character's utterance, without any character names or extra text. The utterance should be a coherent piece of English text.
"""

COUPLE_DIALOGUE_TURN_PROMPT_TEMPLATE = """
Current Role: {current_speaker_name} ({current_speaker_id})
Your Character Profile:
- Guessed Gender: {current_speaker_persona.guessed_gender}
- Appearance Description: {current_speaker_persona.visual_description}
- Fictional Traits: {fictional_traits_str_current}
{current_speaker_birthday_info}
{current_speaker_hobbies_info}
{current_speaker_personality_info}

Conversation Partner: {other_speaker_name} ({other_speaker_id})
Partner's Character Profile:
- Guessed Gender: {other_speaker_persona.guessed_gender}
- Appearance Description: {other_speaker_persona.visual_description}
- Fictional Traits: {fictional_traits_str_other}
{other_speaker_birthday_info}
{other_speaker_hobbies_info}
{other_speaker_personality_info}

Current Dialogue Topic: "{current_topic}"

{conversation_history_segment}
Now, please speak as {current_speaker_name}.
Your utterance:
"""
# --- LangChain LLMs and Parsers ---
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("CRITICAL: GOOGLE_API_KEY environment variable not set.")
    
# multimodal_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.7,
#     generation_config={"response_mime_type": "application/json"}
# )
multimodal_llm =ChatTongyi(
    temperature=0.7,
    generation_config={"response_mime_type": "application/json"},
)
# text_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.7,
#     generation_config={"response_mime_type": "application/json"}
# )
text_llm = ChatTongyi(
    temperature=0.7,
    generation_config={"response_mime_type": "application/json"}
)

image_analysis_parser = JsonOutputParser(pydantic_object=ImageAnalysis)
compatibility_parser = JsonOutputParser(pydantic_object=CompatibilityResult)
dialogue_utterance_parser = StrOutputParser()

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
    chain = multimodal_llm | JsonOutputParser()

    try:
        result_raw = await chain.ainvoke([human_message])
        logger.info(f"Completed visual analysis for person {person_identifier}.")
        return result_raw
    except Exception as e:
        logger.error(f"Image analysis error for person {person_identifier}: {e}", exc_info=True)
        raise

async def calculate_compatibility_async(
    analysis1_visual_data: dict, provided_data1: dict, full_analysis1: ImageAnalysis,
    analysis2_visual_data: dict, provided_data2: dict, full_analysis2: ImageAnalysis,
    dialogue_result: CoupleDialogueResult
) -> CompatibilityResult:
    logger.info("Calculating couple compatibility with dialogue context...")
    prompt_template = ChatPromptTemplate.from_template(COMPATIBILITY_PROMPT_TEXT)
    chain = prompt_template | text_llm | compatibility_parser

    analysis1_visual_json_str = jsonify(analysis1_visual_data).get_data(as_text=True)
    analysis2_visual_json_str = jsonify(analysis2_visual_data).get_data(as_text=True)

    dialogue_summary_lines = []
    if dialogue_result and dialogue_result.dialogues_by_topic:
        for topic_dialogue in dialogue_result.dialogues_by_topic:
            dialogue_summary_lines.append(f"Topic: {topic_dialogue.topic}")
            for turn in topic_dialogue.dialogue_history:
                dialogue_summary_lines.append(f"{turn.speaker_name}: {turn.utterance}")
            dialogue_summary_lines.append("-" * 20)
    dialogue_summary = "\n".join(dialogue_summary_lines)
    if not dialogue_summary:
        dialogue_summary = "No simulated dialogue content."

    try:
        result = await chain.ainvoke({
            "analysis1_visual_json_str": analysis1_visual_json_str,
            "provided_gender1": provided_data1.get('gender', 'not provided'),
            "provided_birthday1": provided_data1.get('birthday', 'not provided'),
            "zodiac_sign1": full_analysis1.zodiac_sign or 'not provided',
            "provided_hobbies1": provided_data1.get('hobbies', 'not provided'),
            "provided_personality1": provided_data1.get('personality', 'not provided'),
            
            "analysis2_visual_json_str": analysis2_visual_json_str,
            "provided_gender2": provided_data2.get('gender', 'not provided'),
            "provided_birthday2": provided_data2.get('birthday', 'not provided'),
            "zodiac_sign2": full_analysis2.zodiac_sign or 'not provided',
            "provided_hobbies2": provided_data2.get('hobbies', 'not provided'),
            "provided_personality2": provided_data2.get('personality', 'not provided'),
            "dialogue_summary": dialogue_summary
        })
        logger.info("Compatibility calculation completed.")
        return result
    except Exception as e:
        logger.error(f"Compatibility calculation error: {e}", exc_info=True)
        raise

async def run_couple_dialogue_async(
    person1_analysis: ImageAnalysis,
    person2_analysis: ImageAnalysis,
    topics: List[str],
    num_turns_per_topic: int = 2,
    person1_name: str = "Person 1",
    person2_name: str = "Person 2"
) -> CoupleDialogueResult:
    """Simulates a dialogue between two AI agents based on their personas and topics."""
    logger.info(f"Starting simulated couple dialogue. Topics: {topics}, Turns per topic: {num_turns_per_topic}")
    
    all_topic_dialogues: List[TopicDialogue] = []

    for topic in topics:
        logger.info(f"Processing topic: {topic}")
        current_dialogue_history_for_topic: List[DialogueTurn] = []
        
        personas = {
            "person1": person1_analysis,
            "person2": person2_analysis,
        }
        speaker_names = {
            "person1": person1_name,
            "person2": person2_name,
        }
        
        current_speaker_id = "person1" 

        for turn in range(num_turns_per_topic * 2):
            other_speaker_id = "person2" if current_speaker_id == "person1" else "person1"
            
            current_speaker_persona = personas[current_speaker_id]
            other_speaker_persona = personas[other_speaker_id]
            
            current_speaker_name_used = speaker_names[current_speaker_id]
            other_speaker_name_used = speaker_names[other_speaker_id]

            fictional_traits_str_current = ", ".join(current_speaker_persona.fictional_bazi_traits)
            fictional_traits_str_other = ", ".join(other_speaker_persona.fictional_bazi_traits)

            current_speaker_birthday_info = f"- Birthday: {current_speaker_persona.provided_birthday} (Zodiac: {current_speaker_persona.zodiac_sign})" if current_speaker_persona.provided_birthday else "- Birthday: Not provided"
            other_speaker_birthday_info = f"- Birthday: {other_speaker_persona.provided_birthday} (Zodiac: {other_speaker_persona.zodiac_sign})" if other_speaker_persona.provided_birthday else "- Birthday: Not provided"
            
            current_speaker_hobbies_info = f"- Hobbies: {', '.join(current_speaker_persona.hobbies)}" if current_speaker_persona.hobbies else "- Hobbies: Not provided"
            other_speaker_hobbies_info = f"- Hobbies: {', '.join(other_speaker_persona.hobbies)}" if other_speaker_persona.hobbies else "- Hobbies: Not provided"

            current_speaker_personality_info = f"- Personality: {current_speaker_persona.personality}" if current_speaker_persona.personality else "- Personality: Not provided"
            other_speaker_personality_info = f"- Personality: {other_speaker_persona.personality}" if other_speaker_persona.personality else "- Personality: Not provided"

            conversation_history_segment = "\n".join(
                [f"{turn.speaker_name}: {turn.utterance}" for turn in current_dialogue_history_for_topic]
            )
            if not conversation_history_segment:
                 conversation_history_segment = "This is the start of the conversation."

            prompt_values = {
                "current_speaker_id": current_speaker_id,
                "current_speaker_name": current_speaker_name_used,
                "current_speaker_persona": current_speaker_persona,
                "fictional_traits_str_current": fictional_traits_str_current,
                "current_speaker_birthday_info": current_speaker_birthday_info,
                "current_speaker_hobbies_info": current_speaker_hobbies_info,
                "current_speaker_personality_info": current_speaker_personality_info,
                "other_speaker_id": other_speaker_id,
                "other_speaker_name": other_speaker_name_used,
                "other_speaker_persona": other_speaker_persona,
                "fictional_traits_str_other": fictional_traits_str_other,
                "other_speaker_birthday_info": other_speaker_birthday_info,
                "other_speaker_hobbies_info": other_speaker_hobbies_info,
                "other_speaker_personality_info": other_speaker_personality_info,
                "current_topic": topic,
                "conversation_history_segment": conversation_history_segment,
            }
            
            chat_prompt_messages = [
                SystemMessage(content=COUPLE_DIALOGUE_SYSTEM_PROMPT),
                HumanMessage(content=COUPLE_DIALOGUE_TURN_PROMPT_TEMPLATE.format(**prompt_values))
            ]
            
            dialogue_chain = ChatPromptTemplate.from_messages(chat_prompt_messages) | text_llm | dialogue_utterance_parser
            
            logger.debug(f"Generating utterance for {current_speaker_name_used}, Topic: {topic}, Turn: {len(current_dialogue_history_for_topic) // 2 + 1}")
            
            try:
                utterance = await dialogue_chain.ainvoke({})
                logger.info(f"{current_speaker_name_used} ({topic}): {utterance}")
            except Exception as e:
                logger.error(f"Error generating utterance for {current_speaker_name_used}: {e}", exc_info=True)
                utterance = f"({current_speaker_name_used} thinking...)"

            current_dialogue_history_for_topic.append(
                DialogueTurn(
                    speaker_id=current_speaker_id,
                    speaker_name=current_speaker_name_used,
                    utterance=utterance.strip()
                )
            )
            
            current_speaker_id = other_speaker_id
            
        all_topic_dialogues.append(
            TopicDialogue(topic=topic, dialogue_history=current_dialogue_history_for_topic)
        )
        logger.info(f"Dialogue for topic '{topic}' completed.")

    return CoupleDialogueResult(
        dialogues_by_topic=all_topic_dialogues,
        person1_name_used=person1_name,
        person2_name_used=person2_name
    )

async def simulate_couple_dialogue(
    person1_analysis: ImageAnalysis,
    person2_analysis: ImageAnalysis, 
    ) -> CoupleDialogueResult:

    logger.debug(f"simulate_couple_dialogue")

    try:
        person1_analysis_data = person1_analysis
        person2_analysis_data = person2_analysis
        
        if not person1_analysis_data or not person2_analysis_data:
            raise ValueError("Missing person analysis data (person1_analysis or person2_analysis)")

        topics = ["Love", "Family", "Future Plans", "Hobbies", "Travel Experiences"]  # Default topics
        
        num_turns_per_topic = 2
        
        person1_name="Person 1"
        person2_name="Person 2"

        logger.info(f"Starting dialogue simulation, Person 1: {person1_name}, Person 2: {person2_name}, Topics: {topics}, Turns per topic: {num_turns_per_topic}")

        dialogue_result = await run_couple_dialogue_async(
            person1_analysis=person1_analysis,
            person2_analysis=person2_analysis,
            topics=topics,
            num_turns_per_topic=num_turns_per_topic,
            person1_name=person1_name,
            person2_name=person2_name
        )
        
        logger.info("Couple dialogue simulation successfully completed.")
        return dialogue_result
    except Exception as e:
        logger.error(f"/api/simulate_couple_dialogue processing error: {type(e).__name__} - {e}", exc_info=True)
        error_message_str = str(e)
        google_api_message = getattr(e, 'message', '') 
        if "User location is not supported" in error_message_str or "User location is not supported" in google_api_message:
             raise ValueError("API service area restriction: User location is not supported.")
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
            analysis1_visual_task = analyze_single_image_async(image1_base64, "1")
            analysis2_visual_task = analyze_single_image_async(image2_base64, "2")
            
            person1_visual_analysis, person2_visual_analysis = await asyncio.gather(
                analysis1_visual_task,
                analysis2_visual_task
            )

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

            person1_full_analysis = ImageAnalysis(
                guessed_gender=person1_visual_analysis.get('guessed_gender', 'undeterminable'),
                visual_description=person1_visual_analysis.get('visual_description', ''),
                fictional_bazi_traits=person1_visual_analysis.get('fictional_bazi_traits', []),
                provided_gender=provided_data1['gender'],
                provided_birthday=provided_data1['birthday'],
                zodiac_sign=zodiac_sign1,
                hobbies=provided_data1['hobbies'],
                personality=provided_data1['personality']
            )
            person2_full_analysis = ImageAnalysis(
                guessed_gender=person2_visual_analysis.get('guessed_gender', 'undeterminable'),
                visual_description=person2_visual_analysis.get('visual_description', ''),
                fictional_bazi_traits=person2_visual_analysis.get('fictional_bazi_traits', []),
                provided_gender=provided_data2['gender'],
                provided_birthday=provided_data2['birthday'],
                zodiac_sign=zodiac_sign2,
                hobbies=provided_data2['hobbies'],
                personality=provided_data2['personality']
            )

            dialogue_result = await simulate_couple_dialogue(
                person1_analysis=person1_full_analysis,
                person2_analysis=person2_full_analysis
            )

            compatibility = await calculate_compatibility_async(
                person1_visual_analysis, provided_data1, person1_full_analysis,
                person2_visual_analysis, provided_data2, person2_full_analysis,
                dialogue_result
            )
            
            return {
                "person1_analysis": person1_full_analysis.dict(),
                "person2_analysis": person2_full_analysis.dict(),
                "compatibility_result": compatibility.dict(),
                "dialogue": dialogue_result.dict()
            }

        result_data = asyncio.run(_run_analysis_pipeline())
        logger.info("/api/analyze_couple request processed successfully.")
        return jsonify(result_data)

    except Exception as e:
        logger.error(f"/api/analyze_couple processing error: {e}", exc_info=True)
        error_message_str = str(e)
        google_api_message = getattr(e, 'message', '')
        if "User location is not supported" in error_message_str or "User location is not supported" in google_api_message:
             return jsonify({"error": "API service area restriction: User location is not supported. Please try using a VPN or deploying the service in a supported region."}), 400
        elif "quota" in error_message_str or "rate limit" in error_message_str:
             return jsonify({"error": "API request rate limit exceeded: Please try again later or check your API quota."}), 429
        elif "authentication" in error_message_str or "API key" in error_message_str or "invalid credential" in error_message_str:
            return jsonify({"error": "Authentication failed: Invalid or expired API key. Please check your GOOGLE_API_KEY and DASHSCOPE_API_KEY."}), 401
        elif "timeout" in error_message_str:
            return jsonify({"error": "API request timed out: Please check network connection or try again later."}), 504
        else:
            return jsonify({"error": f"Internal server error: {e}"}), 500

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