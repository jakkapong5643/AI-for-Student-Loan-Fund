
import json
from utils.langchain_helpers import call_llm
import logging

logger = logging.getLogger(__name__)

import re

def extract_json_from_text(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else None


def plan_questions(text: str) -> dict:

    prompt = f"""
ข้อความต่อไปนี้เป็นข้อความภาษาไทยที่ผ่านการตรวจสอบความถูกต้องแล้ว:

{text}

โปรดวางแผนการสร้างคำถามสำหรับข้อความนี้ โดยระบุข้อมูลดังนี้:

1. จำนวนคำถามทั้งหมดที่เหมาะสม (num_questions)
2. สัดส่วนจำนวนคำถามในแต่ละระดับความยาก ได้แก่ easy, medium, hard (distribution)
3. ประเภทคำตอบที่เหมาะสมสำหรับแต่ละระดับความยาก ได้แก่ "short" หรือ "long" (answer_types)

กรุณาตอบกลับในรูปแบบ JSON เท่านั้น ตามตัวอย่างนี้:

{{
  "num_questions": 5,
  "distribution": {{
    "easy": 2,
    "medium": 2,
    "hard": 1
  }},
  "answer_types": {{
    "easy": "short",
    "medium": "short",
    "hard": "long"
  }}
}}

ห้ามเพิ่มข้อมูลหรือคำอธิบายอื่น ๆ นอกเหนือจาก JSON นี้
"""

    response = call_llm(prompt)

    json_str = extract_json_from_text(response)

    try:
        plan = json.loads(json_str)
        if not all(k in plan for k in ["num_questions", "distribution", "answer_types"]):
            raise ValueError("Response JSON missing keys")
    except Exception as e:
        logger.warning(f"ใช้แผน default แทน เพราะ LLM response มีปัญหา: {e}")
        plan = {
            "num_questions": 15,
            "distribution": {"easy": 5, "medium": 5, "hard": 5},
            "answer_types": {"easy": "short", "medium": "short", "hard": "long"}
        }

    return plan
