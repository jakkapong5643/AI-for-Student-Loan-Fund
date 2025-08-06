from utils.langchain_helpers import call_llm
from utils.thai_text_utils import normalize_thai_text

def clean_text(raw_text: str, feedback_reason: str = "") -> str:
    normalized_text = normalize_thai_text(raw_text)

    # ใส่ feedback เข้ามาอย่างชัดเจนถ้ามี
    feedback_note = f"""
ข้อเสนอแนะจากรอบก่อน:
{feedback_reason}

กรุณานำข้อเสนอแนะด้านบนไปใช้ในการปรับปรุงข้อความที่สะอาดและชัดเจนขึ้น
""" if feedback_reason else ""

    prompt = f"""นี่คือข้อความ OCR ภาษาไทยที่อาจมีข้อผิดพลาด:

{normalized_text}

{feedback_note}

โปรดแก้ไขข้อความนี้โดยทำตามข้อกำหนดต่อไปนี้:

1. แก้ไขข้อผิดพลาดของภาษาไทยในข้อความ เช่น การสะกดคำผิด หรือตัวอักษรผิด
2. ไม่เพิ่มเนื้อหาใหม่ หรือเปลี่ยนความหมายของข้อความเดิม
3. ทำให้ข้อความอ่านเข้าใจง่ายและลื่นไหลขึ้น
4. ตอบกลับเฉพาะข้อความที่แก้ไขแล้วเท่านั้น โดยไม่ต้องมีคำอธิบายหรือข้อความเพิ่มเติมใด ๆ

โปรดเริ่มแก้ไขข้อความจากบรรทัดนี้:"""

    cleaned_text = call_llm(prompt)
    return cleaned_text
