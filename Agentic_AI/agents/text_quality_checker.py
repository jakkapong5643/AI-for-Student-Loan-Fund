from utils.langchain_helpers import call_openai_llm
from configs import settings

def evaluate_text_quality(text: str) -> int:
    """
    ให้คะแนนคุณภาพข้อความ (1-10) โดย LLM ประเมินความสมเหตุสมผล, ครบถ้วน, ความยาวเหมาะสม
    """
    prompt = f"""
ข้อความต่อไปนี้เป็นข้อความภาษาไทยที่ได้รับการแก้ไขแล้ว:

{text}

กรุณาให้คะแนนคุณภาพของข้อความนี้เป็นจำนวนเต็ม 1 ถึง 10 โดยที่  
- 10 หมายถึงคุณภาพดีที่สุด  
- 1 หมายถึงคุณภาพแย่มาก  

โปรดพิจารณาความสมเหตุสมผลและความครบถ้วนของเนื้อหาอย่างเคร่งครัด

**ตอบกลับเป็นตัวเลขจำนวนเต็มตัวเดียวเท่านั้น ห้ามมีข้อความอื่นใด**
"""

    try:
        score_str = call_openai_llm(prompt)
        score = int(score_str.strip())
        if score < 1: score = 1
        if score > 10: score = 10
        return score
    except Exception:
        return 1  # กรณีผิดพลาด ให้คะแนนต่ำสุด
