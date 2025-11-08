from utils.langchain_helpers import call_llm
from configs import settings
# evaluate text LLM as a judge
def evaluate_text_quality(text: str) -> int:
    prompt = f"""
ข้อความภาษาไทยต่อไปนี้ได้รับการแก้ไขแล้ว:

{text}

โปรดให้คะแนนคุณภาพของข้อความนี้ในระดับ 1 ถึง 5 โดยพิจารณาจาก
- ความสมเหตุสมผลของเนื้อหา
- ความถูกต้องของภาษาและไวยากรณ์
- ความชัดเจนและความลื่นไหลในการสื่อสาร

ถ้ามีข้อผิดพลาดเล็กน้อยที่ไม่ร้ายแรง เช่น คำสับสนเล็กน้อย หรือประโยคที่ไม่ลื่นไหลมากนัก ให้คะแนนสูงกว่ากลางได้ (เช่น 3 หรือ 4)

**ตอบกลับด้วยตัวเลขจำนวนเต็มตัวเดียว 1, 2, 3, 4 หรือ 5 เท่านั้น(โดยที่ 5 คือดีที่สุด และ 1 คือแย่มาก) ห้ามมีข้อความอื่นใด**
"""
    try:
        response = call_llm(prompt)
        score_str = response.strip()
        score = int(score_str)
        if score < 1: score = 1
        if score > 5: score = 5
        return score
    except Exception:
        return 0
# Get Feedback reason
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

