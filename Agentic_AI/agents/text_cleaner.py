from utils.langchain_helpers import call_llm
from utils.thai_text_utils import normalize_thai_text

def clean_text(raw_text: str) -> str:
    normalized_text = normalize_thai_text(raw_text)
    prompt = f"""นี่คือข้อความ OCR ภาษาไทยที่อาจมีข้อผิดพลาด:

{normalized_text}

โปรดแก้ไขข้อความนี้โดยทำตามข้อกำหนดต่อไปนี้:

1. แก้ไขข้อผิดพลาดของภาษาไทยในข้อความ เช่น การสะกดคำผิด คำซ้ำ หรือตัวอักษรผิด
2. ไม่เพิ่มเนื้อหาใหม่ หรือเปลี่ยนความหมายของข้อความเดิม
3. ทำให้ข้อความอ่านเข้าใจง่ายและลื่นไหลขึ้น
4. ตอบกลับเฉพาะข้อความที่แก้ไขแล้วเท่านั้น โดยไม่ต้องมีคำอธิบายหรือข้อความเพิ่มเติมใด ๆ

โปรดเริ่มแก้ไขข้อความจากบรรทัดนี้:"""
    cleaned_text = call_llm(prompt)
    return cleaned_text
