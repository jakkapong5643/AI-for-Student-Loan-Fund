from utils.langchain_helpers import call_llm
from configs import settings

def evaluate_text_quality(text: str) -> int:

    prompt = f"""
ข้อความต่อไปนี้เป็นข้อความภาษาไทยที่ได้รับการแก้ไขแล้ว:

{text}

กรุณาให้คะแนนคุณภาพของข้อความนี้เป็นจำนวนเต็ม 1 ถึง 5 โดยที่  
- 5 หมายถึงคุณภาพดีที่สุด  
- 1 หมายถึงคุณภาพแย่มาก  

โปรดพิจารณาความสมเหตุสมผลและถูกต้องของข้อความ

**ตอบกลับเป็นตัวเลขจำนวนเต็มตัวเดียวเท่านั้น ห้ามมีข้อความอื่นใด**
"""

    try:
        score_str = call_llm(prompt)
        score = int(score_str.strip())
        if score < 1: score = 1
        if score > 5: score = 5
        return score
    except Exception:
        return 0
    
def get_feedback_reason(text: str) -> str:
    prompt = f"""
ข้อความต่อไปนี้เป็นข้อความภาษาไทยที่ได้รับการแก้ไขแล้ว:

{text}

กรุณาวิเคราะห์และบอกข้อผิดพลาดหรือข้อบกพร่องของข้อความนี้ โดยเน้นที่จุดที่ทำให้ข้อความนี้มีคุณภาพไม่ดี เช่น ความไม่สมเหตุสมผล ข้อผิดพลาดทางภาษา หรือจุดที่ควรปรับปรุง

กรุณาตอบกลับเป็นข้อความสั้น ๆ ที่อธิบายปัญหาหรือข้อเสนอแนะที่ชัดเจนและตรงประเด็นเท่านั้น (ไม่ต้องเขียนคำอธิบายเพิ่มเติม)
"""
    try:
        feedback = call_llm(prompt)
        return feedback.strip()
    except Exception as e:
        return "ไม่สามารถวิเคราะห์เหตุผลได้ในขณะนี้"
