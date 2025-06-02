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
    os.environ["GOOGLE_API_KEY"]="AIzaSyDpPurzES8YtReAZJEsXcld2Nnfgc6mT94" # 请替换为您的真实API密钥
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = "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" # 请替换为您的真实API密钥
    os.environ["LANGSMITH_PROJECT"] = "AI Couple Compatibility Analyzer"
    os.environ["DASHSCOPE_API_KEY"] = 'sk-c7bcac6af25d4fe8ad9907a1049b1363'
init()
# --- 加载环境变量 ---
load_dotenv()

# --- LangSmith 配置 ---
app = Flask(__name__)
CORS(app)

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 星座辅助函数 ---
def get_zodiac_sign(month: int, day: int) -> str:
    if (month == 1 and day >= 20) or (month == 2 and day <= 18):
        return "Aquarius"  # 水瓶座
    elif (month == 2 and day >= 19) or (month == 3 and day <= 20):
        return "Pisces"  # 双鱼座
    elif (month == 3 and day >= 21) or (month == 4 and day <= 19):
        return "Aries"  # 白羊座
    elif (month == 4 and day >= 20) or (month == 5 and day <= 20):
        return "Taurus"  # 金牛座
    elif (month == 5 and day >= 21) or (month == 6 and day <= 21):
        return "Gemini"  # 双子座
    elif (month == 6 and day >= 22) or (month == 7 and day <= 22):
        return "Cancer"  # 巨蟹座
    elif (month == 7 and day >= 23) or (month == 8 and day <= 22):
        return "Leo"  # 狮子座
    elif (month == 8 and day >= 23) or (month == 9 and day <= 22):
        return "Virgo"  # 处女座
    elif (month == 9 and day >= 23) or (month == 10 and day <= 23):
        return "Libra"  # 天秤座
    elif (month == 10 and day >= 24) or (month == 11 and day <= 22):
        return "Scorpio"  # 天蝎座
    elif (month == 11 and day >= 23) or (month == 12 and day <= 21):
        return "Sagittarius"  # 射手座
    elif (month == 12 and day >= 22) or (month == 1 and day <= 19):
        return "Capricorn"  # 摩羯座
    return "Unknown Zodiac Sign" # 未知星座

# --- 用于结构化输出的 Pydantic 模型 ---
class ImageAnalysis(BaseModel):
    guessed_gender: str = Field(description="基于视觉线索猜测的性别（例如：男性、女性、无法确定）")
    visual_description: str = Field(description="对外貌和风度的简要描述")
    fictional_bazi_traits: List[str] = Field(description="基于视觉外观虚构的元素特征（仅供娱乐）")
    provided_gender: Optional[str] = Field(default=None, description="用户提供的性别。")
    provided_birthday: Optional[str] = Field(default=None, description="用户提供的生日（YYYY-MM-DD格式）。")
    zodiac_sign: Optional[str] = Field(default=None, description="根据提供的生日推断出的星座。")
    hobbies: Optional[List[str]] = Field(default=None, description="用户提供的爱好。")
    personality: Optional[str] = Field(default=None, description="用户提供的性格特征。")

class CompatibilityResult(BaseModel):
    """表示情侣兼容性结果。"""
    compatibility_score: int = Field(description="兼容性得分（0-100）")
    compatibility_explanation: str = Field(description="兼容性得分的解释，考虑视觉特征、提供的信息和模拟对话（仅供娱乐）")
    dialogue_compatibility_assessment: Optional[str] = Field(default=None, description="基于模拟对话的兼容性评估。")

class DialogueTurn(BaseModel):
    speaker_id: str = Field(description="说话者ID（'person1' 或 'person2'）")
    speaker_name: str = Field(description="说话者姓名（例如：'人物1' 或 '人物2'）")
    utterance: str = Field(description="发言内容")

class TopicDialogue(BaseModel):
    topic: str = Field(description="对话主题")
    dialogue_history: List[DialogueTurn] = Field(description="此主题的对话历史")

class CoupleDialogueResult(BaseModel):
    dialogues_by_topic: List[TopicDialogue] = Field(description="按主题组织的完整对话结果")
    person1_name_used: str = Field(default="人物1")
    person2_name_used: str = Field(default="人物2")

# --- Prompts ---
IMAGE_ANALYSIS_PROMPT_TEXT = """
你是一个中立、基于事实且精确的AI助手。请分析图片中的人物。
仅根据单张图片中人物的视觉外观和神态，提供一个有效的JSON对象，包含以下键：

"guessed_gender": 仅根据视觉线索判断的人物性别（例如："男性"，"女性"，"无法确定"）。
"visual_description": 对人物外貌、任何明显情绪或个性特征的简洁、中立和尊重的描述（英文）。
"fictional_bazi_traits": 仅供娱乐，富有创意地、比喻性地虚构2-3个受中国古代五行（木、火、土、金、水）启发的“元素”或“个性”特征。这些特征纯粹基于人物的视觉印象或感知到的气场（英文）。这些虚构特征应纯属想象，不基于真实的出生数据或算命。
输出有效的JSON对象，格式如下：{
  "guessed_gender": "字符串",
  "visual_description": "字符串",
  "fictional_bazi_traits": ["字符串", "字符串", "字符串"]
}
"""

COMPATIBILITY_PROMPT_TEXT = """
你是一个幽默且富有创意的AI助手。你将收到两个人的分析结果，以及他们之间的一段模拟对话。
请根据以下所有信息，提供一份情侣兼容性报告：

人物1的详细信息：
- 视觉分析：{analysis1_visual_json_str}
- 提供的信息：
  - 性别：{provided_gender1}
  - 生日：{provided_birthday1}
  - 星座：{zodiac_sign1}
  - 爱好：{provided_hobbies1}
  - 性格：{provided_personality1}

人物2的详细信息：
- 视觉分析：{analysis2_visual_json_str}
- 提供的信息：
  - 性别：{provided_gender2}
  - 生日：{provided_birthday2}
  - 星座：{zodiac_sign2}
  - 爱好：{provided_hobbies2}
  - 性格：{provided_personality2}

模拟对话内容：
{dialogue_summary}

根据以上所有信息（AI的视觉分析、用户提供的信息、星座、爱好、性格以及模拟对话内容），生成：
1. 一个0-100之间的“兼容性得分”。
2. 一段简短、积极且有趣的解释（compatibility_explanation），重点说明他们结合的特质和兴趣如何互补，并反映对话中显示的融洽关系。
3. 基于模拟对话内容的评估（dialogue_compatibility_assessment），解释对话如何反映他们的兼容性，例如他们的互动风格、共同兴趣的发现等。

请强调所有结果仅供创意娱乐，不构成真实的关系建议。

输出包含以下键的JSON：
{{
  "compatibility_score": 整数,
  "compatibility_explanation": "字符串",
  "dialogue_compatibility_assessment": "字符串"
}}
"""

COUPLE_DIALOGUE_SYSTEM_PROMPT = """你将扮演两个角色中的一个，并围绕指定主题进行轻松愉快的浪漫对话。
请根据你的角色设定和提供的对话历史进行回应。你的目标是使对话自然、有趣，并表达角色的个性和相互间的情感。
仅输出你当前角色的发言，不要包含任何角色名称或多余的文本。发言内容应为一段连贯的中文文本。
"""

COUPLE_DIALOGUE_TURN_PROMPT_TEMPLATE = """
当前角色：{current_speaker_name} ({current_speaker_id})
你的角色设定：
- 猜测性别：{current_speaker_persona.guessed_gender}
- 外貌描述：{current_speaker_persona.visual_description}
- 虚构特质：{fictional_traits_str_current}
{current_speaker_birthday_info}
{current_speaker_hobbies_info}
{current_speaker_personality_info}

对话伙伴：{other_speaker_name} ({other_speaker_id})
伙伴的角色设定：
- 猜测性别：{other_speaker_persona.guessed_gender}
- 外貌描述：{other_speaker_persona.visual_description}
- 虚构特质：{fictional_traits_str_other}
{other_speaker_birthday_info}
{other_speaker_hobbies_info}
{other_speaker_personality_info}

当前对话主题："{current_topic}"

{conversation_history_segment}
现在，请以{current_speaker_name}的身份发言。
你的发言：
"""
# --- LangChain LLMs and Parsers ---
if not os.getenv("GOOGLE_API_KEY"):
    logger.error("CRITICAL: GOOGLE_API_KEY environment variable not set.") # 严重错误：未设置 GOOGLE_API_KEY 环境变量。
    
# multimodal_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.7,
#     generation_config={"response_mime_type": "application/json"}
# )
multimodal_llm =ChatTongyi( # 多模态LLM
    temperature=0.7,
    generation_config={"response_mime_type": "application/json"},
)
# text_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.7,
#     generation_config={"response_mime_type": "application/json"}
# )
text_llm = ChatTongyi( # 文本LLM
    temperature=0.7,
    generation_config={"response_mime_type": "application/json"}
)

image_analysis_parser = JsonOutputParser(pydantic_object=ImageAnalysis) # 图像分析解析器
compatibility_parser = JsonOutputParser(pydantic_object=CompatibilityResult) # 兼容性解析器
dialogue_utterance_parser = StrOutputParser() # 对话语句解析器

# --- 核心逻辑函数 ---
async def analyze_single_image_async(image_base64: str, person_identifier: str) -> dict:
    logger.info(f"Analyzing image for person {person_identifier}...") # 正在分析人物 {person_identifier} 的图像...
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
        logger.info(f"Completed visual analysis for person {person_identifier}.") # 人物 {person_identifier} 的视觉分析已完成。
        return result_raw
    except Exception as e:
        logger.error(f"Image analysis error for person {person_identifier}: {e}", exc_info=True) # 人物 {person_identifier} 的图像分析错误：{e}
        raise

async def calculate_compatibility_async(
    analysis1_visual_data: dict, provided_data1: dict, full_analysis1: ImageAnalysis,
    analysis2_visual_data: dict, provided_data2: dict, full_analysis2: ImageAnalysis,
    dialogue_result: CoupleDialogueResult
) -> CompatibilityResult:
    logger.info("Calculating couple compatibility with dialogue context...") # 正在结合对话上下文计算情侣兼容性...
    prompt_template = ChatPromptTemplate.from_template(COMPATIBILITY_PROMPT_TEXT)
    chain = prompt_template | text_llm | compatibility_parser

    analysis1_visual_json_str = jsonify(analysis1_visual_data).get_data(as_text=True)
    analysis2_visual_json_str = jsonify(analysis2_visual_data).get_data(as_text=True)

    dialogue_summary_lines = []
    if dialogue_result and dialogue_result.dialogues_by_topic:
        for topic_dialogue in dialogue_result.dialogues_by_topic:
            dialogue_summary_lines.append(f"主题：{topic_dialogue.topic}")
            for turn in topic_dialogue.dialogue_history:
                dialogue_summary_lines.append(f"{turn.speaker_name}：{turn.utterance}")
            dialogue_summary_lines.append("-" * 20)
    dialogue_summary = "\n".join(dialogue_summary_lines)
    if not dialogue_summary:
        dialogue_summary = "无模拟对话内容。" # No simulated dialogue content.

    try:
        result = await chain.ainvoke({
            "analysis1_visual_json_str": analysis1_visual_json_str,
            "provided_gender1": provided_data1.get('gender', '未提供'), # not provided
            "provided_birthday1": provided_data1.get('birthday', '未提供'), # not provided
            "zodiac_sign1": full_analysis1.zodiac_sign or '未提供', # not provided
            "provided_hobbies1": provided_data1.get('hobbies', '未提供'), # not provided
            "provided_personality1": provided_data1.get('personality', '未提供'), # not provided
            
            "analysis2_visual_json_str": analysis2_visual_json_str,
            "provided_gender2": provided_data2.get('gender', '未提供'), # not provided
            "provided_birthday2": provided_data2.get('birthday', '未提供'), # not provided
            "zodiac_sign2": full_analysis2.zodiac_sign or '未提供', # not provided
            "provided_hobbies2": provided_data2.get('hobbies', '未提供'), # not provided
            "provided_personality2": provided_data2.get('personality', '未提供'), # not provided
            "dialogue_summary": dialogue_summary
        })
        logger.info("Compatibility calculation completed.") # 兼容性计算已完成。
        return result
    except Exception as e:
        logger.error(f"Compatibility calculation error: {e}", exc_info=True) # 兼容性计算错误：{e}
        raise

async def run_couple_dialogue_async(
    person1_analysis: ImageAnalysis,
    person2_analysis: ImageAnalysis,
    topics: List[str],
    num_turns_per_topic: int = 2,
    person1_name: str = "人物1", # Person 1
    person2_name: str = "人物2"  # Person 2
) -> CoupleDialogueResult:
    """根据人物角色和主题模拟两个AI智能体之间的对话。"""
    logger.info(f"Starting simulated couple dialogue. Topics: {topics}, Turns per topic: {num_turns_per_topic}") # 开始模拟情侣对话。主题：{topics}，每个主题的回合数：{num_turns_per_topic}
    
    all_topic_dialogues: List[TopicDialogue] = []

    for topic in topics:
        logger.info(f"Processing topic: {topic}") # 正在处理主题：{topic}
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

            current_speaker_gender_info = f"- 性别：{current_speaker_persona.provided_gender}" if current_speaker_persona.provided_gender else "- 性别：未提供" # - Gender: Not provided
            other_speaker_gender_info = f"- 性别：{other_speaker_persona.provided_gender}" if other_speaker_persona.provided_gender else "- 性别：未提供" # - Gender: Not provided
            
            current_speaker_birthday_info = f"- 生日：{current_speaker_persona.provided_birthday} (星座：{current_speaker_persona.zodiac_sign})" if current_speaker_persona.provided_birthday else "- 生日：未提供" # - Birthday: Not provided
            other_speaker_birthday_info = f"- 生日：{other_speaker_persona.provided_birthday} (星座：{other_speaker_persona.zodiac_sign})" if other_speaker_persona.provided_birthday else "- 生日：未提供" # - Birthday: Not provided
            
            current_speaker_hobbies_info = f"- 爱好：{', '.join(current_speaker_persona.hobbies)}" if current_speaker_persona.hobbies else "- 爱好：未提供" # - Hobbies: Not provided
            other_speaker_hobbies_info = f"- 爱好：{', '.join(other_speaker_persona.hobbies)}" if other_speaker_persona.hobbies else "- 爱好：未提供" # - Hobbies: Not provided

            current_speaker_personality_info = f"- 性格：{current_speaker_persona.personality}" if current_speaker_persona.personality else "- 性格：未提供" # - Personality: Not provided
            other_speaker_personality_info = f"- 性格：{other_speaker_persona.personality}" if other_speaker_persona.personality else "- 性格：未提供" # - Personality: Not provided


            conversation_history_segment = "\n".join(
                [f"{turn.speaker_name}：{turn.utterance}" for turn in current_dialogue_history_for_topic]
            )
            if not conversation_history_segment:
                 conversation_history_segment = "这是对话的开始。" # This is the start of the conversation.

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
            
            logger.debug(f"Generating utterance for {current_speaker_name_used}, Topic: {topic}, Turn: {len(current_dialogue_history_for_topic) // 2 + 1}") # 正在为 {current_speaker_name_used} 生成发言，主题：{topic}，回合：{len(current_dialogue_history_for_topic) // 2 + 1}
            
            try:
                utterance = await dialogue_chain.ainvoke({})
                logger.info(f"{current_speaker_name_used} ({topic})：{utterance}")
            except Exception as e:
                logger.error(f"Error generating utterance for {current_speaker_name_used}: {e}", exc_info=True) # 为 {current_speaker_name_used} 生成发言时出错：{e}
                utterance = f"({current_speaker_name_used} 正在思考...)" # ({current_speaker_name_used} thinking...)

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
        logger.info(f"Dialogue for topic '{topic}' completed.") # 主题 '{topic}' 的对话已完成。

    return CoupleDialogueResult(
        dialogues_by_topic=all_topic_dialogues,
        person1_name_used=person1_name,
        person2_name_used=person2_name
    )

async def simulate_couple_dialogue(
    person1_analysis: ImageAnalysis,
    person2_analysis: ImageAnalysis, 
    ) -> CoupleDialogueResult:

    logger.debug(f"simulate_couple_dialogue") # 模拟情侣对话

    try:
        person1_analysis_data = person1_analysis
        person2_analysis_data = person2_analysis
        
        if not person1_analysis_data or not person2_analysis_data:
            raise ValueError("缺少人物分析数据 (person1_analysis 或 person2_analysis)") # Missing person analysis data (person1_analysis or person2_analysis)

        # topics = ["爱情", "家庭", "未来计划", "爱好", "旅行经历"]  # 默认主题 Love, Family, Future Plans, Hobbies, Travel Experiences
        topics = ["个人背景与家庭",
                "价值观与信仰",
                "兴趣爱好与休闲活动",
                "情感需求与沟通方式",
                "未来规划与生活方式",
                "个人成长与自我认知",
                "社交与人际关系",
                "性与亲密关系",
                "文化与传统",
                "幽默与乐趣",
                "对对方的评价"
                ]
        num_turns_per_topic = 2
        
        person1_name="人物1" # Person 1
        person2_name="人物2" # Person 2

        logger.info(f"Starting dialogue simulation, Person 1: {person1_name}, Person 2: {person2_name}, Topics: {topics}, Turns per topic: {num_turns_per_topic}") # 开始对话模拟，人物1：{person1_name}，人物2：{person2_name}，主题：{topics}，每个主题的回合数：{num_turns_per_topic}

        dialogue_result = await run_couple_dialogue_async(
            person1_analysis=person1_analysis,
            person2_analysis=person2_analysis,
            topics=topics,
            num_turns_per_topic=num_turns_per_topic,
            person1_name=person1_name,
            person2_name=person2_name
        )
        
        logger.info("Couple dialogue simulation successfully completed.") # 情侣对话模拟成功完成。
        return dialogue_result
    except Exception as e:
        logger.error(f"/api/simulate_couple_dialogue processing error: {type(e).__name__} - {e}", exc_info=True) # /api/simulate_couple_dialogue 处理错误：{type(e).__name__} - {e}
        error_message_str = str(e)
        google_api_message = getattr(e, 'message', '') 
        if "User location is not supported" in error_message_str or "User location is not supported" in google_api_message:
             raise ValueError("API服务区域限制：不支持用户所在位置。") # API service area restriction: User location is not supported.
        raise


# --- API 端点 ---
@app.route('/api/analyze_couple', methods=['POST'])
def analyze_couple_endpoint():
    if not request.is_json:
        logger.warning("Request was not JSON.") # 请求不是JSON格式。
        return jsonify({"error": "请求必须是JSON格式"}), 400 # Request must be JSON

    data = request.get_json()
    image1_base64 = data.get('image1_base64')
    image2_base64 = data.get('image2_base64')

    # 提取用户提供的数据
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
        logger.warning("Missing image data in request.") # 请求中缺少图像数据。
        return jsonify({"error": "缺少 image1_base64 或 image2_base64"}), 400 # Missing image1_base64 or image2_base64

    logger.info("Received /api/analyze_couple request...") # 收到 /api/analyze_couple 请求...

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
                    logger.warning(f"Invalid birthday format for person 1: {provided_data1['birthday']}") # 人物1的生日格式无效：{provided_data1['birthday']}
            
            zodiac_sign2 = None
            if provided_data2['birthday']:
                try:
                    bday2 = datetime.strptime(provided_data2['birthday'], '%Y-%m-%d')
                    zodiac_sign2 = get_zodiac_sign(bday2.month, bday2.day)
                except ValueError:
                    logger.warning(f"Invalid birthday format for person 2: {provided_data2['birthday']}") # 人物2的生日格式无效：{provided_data2['birthday']}

            person1_full_analysis = ImageAnalysis(
                guessed_gender=person1_visual_analysis.get('guessed_gender', '无法确定'), # undeterminable
                visual_description=person1_visual_analysis.get('visual_description', ''),
                fictional_bazi_traits=person1_visual_analysis.get('fictional_bazi_traits', []),
                provided_gender=provided_data1['gender'],
                provided_birthday=provided_data1['birthday'],
                zodiac_sign=zodiac_sign1,
                hobbies=provided_data1['hobbies'],
                personality=provided_data1['personality']
            )
            person2_full_analysis = ImageAnalysis(
                guessed_gender=person2_visual_analysis.get('guessed_gender', '无法确定'), # undeterminable
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
                "compatibility_result": compatibility,
                "dialogue": dialogue_result.dict()
            }

        result_data = asyncio.run(_run_analysis_pipeline())
        logger.info("/api/analyze_couple request processed successfully.") # /api/analyze_couple 请求处理成功。
        return jsonify(result_data)

    except Exception as e:
        logger.error(f"/api/analyze_couple processing error: {e}", exc_info=True) # /api/analyze_couple 处理错误：{e}
        error_message_str = str(e)
        google_api_message = getattr(e, 'message', '')
        if "User location is not supported" in error_message_str or "User location is not supported" in google_api_message:
             return jsonify({"error": "API服务区域限制：不支持用户所在位置。请尝试使用VPN或在支持的区域部署服务。"}), 400 # API service area restriction: User location is not supported. Please try using a VPN or deploying the service in a supported region.
        elif "quota" in error_message_str or "rate limit" in error_message_str:
             return jsonify({"error": "API请求速率限制超出：请稍后重试或检查您的API配额。"}), 429 # API request rate limit exceeded: Please try again later or check your API quota.
        elif "authentication" in error_message_str or "API key" in error_message_str or "invalid credential" in error_message_str:
            return jsonify({"error": "身份验证失败：API密钥无效或已过期。请检查您的GOOGLE_API_KEY和DASHSCOPE_API_KEY。"}), 401 # Authentication failed: Invalid or expired API key. Please check your GOOGLE_API_KEY and DASHSCOPE_API_KEY.
        elif "timeout" in error_message_str:
            return jsonify({"error": "API请求超时：请检查网络连接或稍后重试。"}), 504 # API request timed out: Please check network connection or try again later.
        else:
            return jsonify({"error": f"内部服务器错误：{e}"}), 500 # Internal server error: {e}

# --- 主执行 ---
if __name__ == '__main__':
    init()
    if not os.getenv("GOOGLE_API_KEY"):
        print("错误：未设置GOOGLE_API_KEY环境变量。") # Error: GOOGLE_API_KEY environment variable not set.
        print("请在您的 .env 文件或环境中进行设置。") # Please set it in your .env file or environment.
    else:
        print("AI情侣兼容性分析器后端正在启动...") # AI Couple Compatibility Analyzer backend starting...
        print("请确保您的GOOGLE_API_KEY和LangSmith（可选）已配置。") # Ensure your GOOGLE_API_KEY and LangSmith (optional) are configured.
        print("服务将在 http://localhost:5001 运行") # Service will run at http://localhost:5001
        app.run(debug=True, port=5001, host='0.0.0.0')
