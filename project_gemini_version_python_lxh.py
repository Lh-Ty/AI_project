# app.py
import os
import asyncio
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage ,AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field ,ValidationError
from langchain_community.chat_models.tongyi import ChatTongyi
def init():
    # os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    # os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'
    os.environ["GOOGLE_API_KEY"]="AIzaSyDpPurzES8YtReAZJEsXcld2Nnfgc6mT94"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = "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"
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

# --- Pydantic Models for Structured Output ---
class ImageAnalysis(BaseModel):
    """Represents the analysis of a single person's image."""
    guessed_gender: str = Field(description="Gender guess (e.g., male, female, undeterminable)")
    visual_description: str = Field(description="Brief description of appearance and demeanor")
    fictional_bazi_traits: list[str] = Field(description="Fictional elemental traits based on visual appearance (for entertainment only)")
    birthday: Optional[str] = Field(default=None, description="Optional birthday information (for entertainment only)")
class CompatibilityResult(BaseModel):
    """Represents the couple compatibility result."""
    compatibility_score: int = Field(description="Compatibility score (0-100)")
    compatibility_explanation: str = Field(description="Explanation of the compatibility score (for entertainment only)")
# 新增：对话模拟相关的Pydantic模型
class DialogueTurn(BaseModel):
    speaker_id: str = Field(description="发言者ID ('person1' 或 'person2')")
    speaker_name: str = Field(description="发言者称呼 (例如 '人物1' 或 '人物2')")
    utterance: str = Field(description="发言内容")

class TopicDialogue(BaseModel):
    topic: str = Field(description="对话主题")
    dialogue_history: List[DialogueTurn] = Field(description="该主题下的对话记录")

class CoupleDialogueResult(BaseModel):
    dialogues_by_topic: List[TopicDialogue] = Field(description="按主题组织的完整对话结果")
    person1_name_used: str = Field(default="人物1")
    person2_name_used: str = Field(default="人物2")
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
You are a humorous and creative AI assistant. You received analysis results for two people based on their images.
Person 1 analysis: {analysis1_json_str}
Person 2 analysis: {analysis2_json_str}

Based on these visual descriptions and purely fictional elemental traits (for entertainment only), generate:
1. A "compatibility score" between 0-100
2. A short, positive, fun explanation focusing on how their visual traits and fictional elements might complement each other.

This is not real relationship advice - just for creative entertainment.

Output JSON with these keys:
{{
  "compatibility_score": integer,
  "compatibility_explanation": "string"
}}
"""
# 新增：情侣对话的提示模板
COUPLE_DIALOGUE_SYSTEM_PROMPT = """你将扮演以下两个角色之一，与另一个人进行一场轻松、浪漫且围绕指定主题的对话。
请根据提供给你的角色设定和对话历史进行回应。你的目标是让对话自然、有趣，并体现出角色的个性和彼此之间的情愫。
请只输出你当前角色的发言内容，不要包含任何角色名称或其他额外文字。发言应为一段连贯的中文文本。
"""

COUPLE_DIALOGUE_TURN_PROMPT_TEMPLATE = """
当前角色: {current_speaker_name} ({current_speaker_id})
你的角色设定:
- 性别猜测: {current_speaker_persona.guessed_gender}
- 外貌神态描述: {current_speaker_persona.visual_description}
- 虚构特征: {fictional_traits_str_current}
{current_speaker_birthday_info}

对话伙伴: {other_speaker_name} ({other_speaker_id})
对话伙伴的角色设定:
- 性别猜测: {other_speaker_persona.guessed_gender}
- 外貌神态描述: {other_speaker_persona.visual_description}
- 虚构特征: {fictional_traits_str_other}
{other_speaker_birthday_info}

当前对话主题: "{current_topic}"

{conversation_history_segment}
现在，请你作为 {current_speaker_name} 发言。
你的发言:
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
dialogue_utterance_parser = StrOutputParser() # 对话的发言是纯字符串
# --- Core Logic Functions ---
async def analyze_single_image_async(image_base64: str, person_identifier: str) -> ImageAnalysis:
    logger.info(f"Analyzing image for person {person_identifier}...")
    message_content = [
        {"type": "text", "text": IMAGE_ANALYSIS_PROMPT_TEXT},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        }
    ]
    human_message = HumanMessage(content=message_content)
    chain = multimodal_llm | image_analysis_parser

    try:
        result = await chain.ainvoke([human_message])
        if isinstance(result, dict):
            result = ImageAnalysis(**result)
        logger.info(f"Completed analysis for person {person_identifier}.")
        return result
    except Exception as e:
        logger.error(f"Image analysis error for person {person_identifier}: {e}", exc_info=True)
        raise

async def calculate_compatibility_async(analysis1: ImageAnalysis, analysis2: ImageAnalysis) -> CompatibilityResult:
    logger.info("Calculating couple compatibility...")
    prompt_template = ChatPromptTemplate.from_template(COMPATIBILITY_PROMPT_TEXT)
    chain = prompt_template | text_llm | compatibility_parser

    try:
        result = await chain.ainvoke({
            "analysis1_json_str": analysis1.json(),
            "analysis2_json_str": analysis2.json()
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
    num_turns_per_topic: int = 2, # 每人每个主题说几句话
    person1_name: str = "人物1",
    person2_name: str = "人物2"
) -> CoupleDialogueResult:
    """Simulates a dialogue between two AI agents based on their personas and topics."""
    logger.info(f"开始模拟情侣对话。主题: {topics}, 每主题轮次: {num_turns_per_topic}")
    
    all_topic_dialogues: List[TopicDialogue] = []

    for topic in topics:
        logger.info(f"正在处理主题: {topic}")
        current_dialogue_history_for_topic: List[DialogueTurn] = []
        
        # 构建角色信息，方便在循环中引用
        personas = {
            "person1": person1_analysis,
            "person2": person2_analysis,
        }
        speaker_names = {
            "person1": person1_name,
            "person2": person2_name,
        }
        
        # 决定谁先发言 (可以随机或固定，这里固定人物1先说)
        current_speaker_id = "person1" 

        for turn in range(num_turns_per_topic * 2): # 总共的发言次数
            other_speaker_id = "person2" if current_speaker_id == "person1" else "person1"
            
            current_speaker_persona = personas[current_speaker_id]
            other_speaker_persona = personas[other_speaker_id]
            
            current_speaker_name_used = speaker_names[current_speaker_id]
            other_speaker_name_used = speaker_names[other_speaker_id]

            # 准备提示模板所需的变量
            fictional_traits_str_current = ", ".join(current_speaker_persona.fictional_bazi_traits)
            fictional_traits_str_other = ", ".join(other_speaker_persona.fictional_bazi_traits)

            current_speaker_birthday_info = f"- 生日: {current_speaker_persona.birthday}" if current_speaker_persona.birthday else "- 生日: 未提供"
            other_speaker_birthday_info = f"- 生日: {other_speaker_persona.birthday}" if other_speaker_persona.birthday else "- 生日: 未提供"

            conversation_history_segment = "\n".join(
                [f"{turn.speaker_name}: {turn.utterance}" for turn in current_dialogue_history_for_topic]
            )
            if not conversation_history_segment:
                 conversation_history_segment = "这是对话的开始。"


            prompt_values = {
                "current_speaker_id": current_speaker_id,
                "current_speaker_name": current_speaker_name_used,
                "current_speaker_persona": current_speaker_persona,
                "fictional_traits_str_current": fictional_traits_str_current,
                "current_speaker_birthday_info": current_speaker_birthday_info,
                "other_speaker_id": other_speaker_id,
                "other_speaker_name": other_speaker_name_used,
                "other_speaker_persona": other_speaker_persona,
                "fictional_traits_str_other": fictional_traits_str_other,
                "other_speaker_birthday_info": other_speaker_birthday_info,
                "current_topic": topic,
                "conversation_history_segment": conversation_history_segment,
            }
            
            # 构建完整的聊天提示
            # SystemMessage 设置整体行为，HumanMessage 包含具体指令和上下文
            chat_prompt_messages = [
                SystemMessage(content=COUPLE_DIALOGUE_SYSTEM_PROMPT),
                HumanMessage(content=COUPLE_DIALOGUE_TURN_PROMPT_TEMPLATE.format(**prompt_values))
            ]
            
            dialogue_chain = ChatPromptTemplate.from_messages(chat_prompt_messages) | text_llm | dialogue_utterance_parser
            
            logger.debug(f"为 {current_speaker_name_used} 生成发言，主题: {topic}, 轮次: {len(current_dialogue_history_for_topic) // 2 + 1}")
            
            try:
                utterance = await dialogue_chain.ainvoke({}) # 因为所有变量都在模板中格式化了
                logger.info(f"{current_speaker_name_used} ({topic}): {utterance}")
            except Exception as e:
                logger.error(f"为 {current_speaker_name_used} 生成发言时出错: {e}", exc_info=True)
                utterance = f"({current_speaker_name_used} 思考中...)" # 发生错误时的默认发言

            current_dialogue_history_for_topic.append(
                DialogueTurn(
                    speaker_id=current_speaker_id,
                    speaker_name=current_speaker_name_used,
                    utterance=utterance.strip()
                )
            )
            
            # 切换发言者
            current_speaker_id = other_speaker_id
            
        all_topic_dialogues.append(
            TopicDialogue(topic=topic, dialogue_history=current_dialogue_history_for_topic)
        )
        logger.info(f"主题 '{topic}' 对话完成。")

    return CoupleDialogueResult(
        dialogues_by_topic=all_topic_dialogues,
        person1_name_used=person1_name,
        person2_name_used=person2_name
    )
# @app.route('/api/simulate_couple_dialogue', methods=['POST', 'OPTIONS'])
async def simulate_couple_dialogue(
    person1_analysis: ImageAnalysis,
    person2_analysis: ImageAnalysis, 
    ) -> CoupleDialogueResult:

    logger.debug(f"simulate_couple_dialogue")

    try:
        # 从请求数据中解析人物分析结果
        # 假设前端会传来符合 ImageAnalysis 结构（或其字典形式）的数据
        person1_analysis_data = person1_analysis
        person2_analysis_data = person2_analysis
        
        if not person1_analysis_data or not person2_analysis_data:
            return jsonify({"error": "缺少人物分析数据 (person1_analysis 或 person2_analysis)"}), 400

        # 将字典转换为Pydantic模型实例
        try:
            # person1_analysis = ImageAnalysis(**person1_analysis_data)
            # person2_analysis = ImageAnalysis(**person2_analysis_data)
            person1_analysis=person1_analysis_data
            person2_analysis=person2_analysis_data
        except ValidationError as ve:
            logger.error(f"Pydantic模型验证失败: {ve}", exc_info=True)
            return jsonify({"error": f"人物分析数据格式错误: {ve}"}), 400
        
        # topics = data.get('topics')
        topics = ["爱情", "家庭", "未来计划", "兴趣爱好", "旅行经历"]  # 默认主题列表
        if not topics or not isinstance(topics, list) or not all(isinstance(t, str) for t in topics):
            return jsonify({"error": "缺少有效的主题列表 (topics)"}), 400
            
        num_turns_per_topic = 2 # 默认为每人2轮
        if not isinstance(num_turns_per_topic, int) or num_turns_per_topic <= 0:
            num_turns_per_topic = 2 # 纠正无效值

        # person1_name = data.get('person1_name', "人物1")
        # person2_name = data.get('person2_name', "人物2")
        person1_name="人物1"
        person2_name="人物2"

        logger.info(f"开始模拟对话，人物1: {person1_name}, 人物2: {person2_name}, 主题: {topics}, 每主题轮次: {num_turns_per_topic}")

        dialogue_result = await run_couple_dialogue_async(
            person1_analysis=person1_analysis,
            person2_analysis=person2_analysis,
            topics=topics,
            num_turns_per_topic=num_turns_per_topic,
            person1_name=person1_name,
            person2_name=person2_name
        )
        
        logger.info("情侣对话模拟成功完成。")
        return dialogue_result.dict()

    except Exception as e:
        logger.error(f"/api/simulate_couple_dialogue 处理过程中发生错误: {type(e).__name__} - {e}", exc_info=True)
        # (错误处理逻辑与之前版本类似，此处省略以减少重复，但实际代码中应保留)
        error_message_str = str(e)
        google_api_message = getattr(e, 'message', '') 
        if "User location is not supported" in error_message_str or "User location is not supported" in google_api_message:
             return jsonify({"error": "API服务区域限制：用户所在位置不受支持。"}), 400
        return jsonify({"error": "对话模拟过程中发生内部服务器错误。"}), 500

# --- API Endpoint ---
@app.route('/api/analyze_couple', methods=['POST'])
def analyze_couple_endpoint():
    if not request.is_json:
        logger.warning("Request was not JSON.")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    image1_base64 = data.get('image1_base64')
    image2_base64 = data.get('image2_base64')
    birthday1=data.get('birthday1')
    birthday2=data.get('birthday2')

    if not image1_base64 or not image2_base64:
        logger.warning("Missing image data in request.")
        return jsonify({"error": "Missing image1_base64 or image2_base64"}), 400

    logger.info("Received /api/analyze_couple request...")

    try:
        async def _run_analysis_pipeline():
            analysis1_task = analyze_single_image_async(image1_base64, "1")
            analysis2_task = analyze_single_image_async(image2_base64, "2")
            
            person1_analysis, person2_analysis = await asyncio.gather(
                analysis1_task,
                analysis2_task
            )
            dialogue=await simulate_couple_dialogue(
                person1_analysis=person1_analysis,
                person2_analysis=person2_analysis
            )
            compatibility = await calculate_compatibility_async(person1_analysis, person2_analysis)
            print(type(dialogue), dialogue)
            return {
                "person1_analysis": person1_analysis.dict(),
                "person2_analysis": person2_analysis.dict(),
                "compatibility_result": compatibility,
                "dialogue": dialogue
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