from utils.langchain_helpers import call_openai_llm

def generate_qa(text: str, plan: dict):
    """
    สร้างชุดคำถาม-คำตอบ ตามแผนที่กำหนด
    คืนค่า list dict: [{'question_text':..., 'answer_text':..., 'difficulty_level':..., 'answer_type':...}, ...]
    """
    total_q = plan.get("num_questions", 5)
    dist = plan.get("distribution", {})
    answer_types = plan.get("answer_types", {})

    qas = []

    prompt = f"""
ข้อความต่อไปนี้เป็นข้อความภาษาไทย:

{text}

กรุณาสร้างชุดคำถาม-คำตอบจำนวนทั้งหมด {total_q} คู่ โดยแบ่งตามระดับความยากและประเภทคำตอบ ดังนี้:

- Easy: {dist.get('easy', 0)} คำถาม (ตอบสั้น)
- Medium: {dist.get('medium', 0)} คำถาม (ตอบสั้น)
- Hard: {dist.get('hard', 0)} คำถาม (ตอบยาว)

โปรดใช้รูปแบบผลลัพธ์ตามนี้สำหรับแต่ละคู่คำถาม-คำตอบ:

คำถาม: <ข้อความคำถาม>
คำตอบ: <ข้อความคำตอบ>
ระดับความยาก: <easy/medium/hard>
ประเภทคำตอบ: <short/long>

หมายเหตุ:
- คำตอบสั้น (short) คือคำตอบที่กระชับ ไม่เกิน 2-3 ประโยค
- คำตอบยาว (long) คือคำตอบที่อธิบายละเอียดและครบถ้วน

กรุณาตอบกลับโดยตรงเริ่มสร้างคำถาม-คำตอบตามรูปแบบนี้ทันที:
"""


    response = call_openai_llm(prompt)

    lines = response.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("คำถาม:"):
            q = line[len("คำถาม:"):].strip()
            qas.append({"question_text": q})
        elif line.startswith("คำตอบ:") and qas:
            a = line[len("คำตอบ:"):].strip()
            qas[-1]["answer_text"] = a
        elif line.startswith("ระดับความยาก:") and qas:
            d = line[len("ระดับความยาก:"):].strip()
            qas[-1]["difficulty_level"] = d
        elif line.startswith("ประเภทคำตอบ:") and qas:
            t = line[len("ประเภทคำตอบ:"):].strip()
            qas[-1]["answer_type"] = t

    return qas
