from utils.langchain_helpers import call_llm

def evaluate(qa_pair: dict) -> dict:

    prompt = f"""
กรุณาประเมินคุณภาพของคำถามและคำตอบต่อไปนี้ โดยให้คะแนนในช่วง 1-10 (โดย 10 คือดีที่สุด และ 1 คือแย่มาก) โดยพิจารณาจาก:

- ความชัดเจนของคำถามและคำตอบ
- ความสอดคล้องกับข้อความต้นทาง
- ความครบถ้วนของข้อมูลในคำตอบ

คำถาม: {qa_pair.get('question_text')}
คำตอบ: {qa_pair.get('answer_text')}

โปรดตอบผลลัพธ์ในรูปแบบ JSON เท่านั้น ตามฟอร์แมตนี้:

{{
  "question_quality_score": <int จาก 1 ถึง 10>,
  "answer_quality_score": <int จาก 1 ถึง 10>,
  "overall_qa_score": <float จาก 1.0 ถึง 10.0>,
  "pass_fail_flag": "<pass หรือ fail>"
}}

ข้อควรระวัง:  
- ตอบเฉพาะ JSON ตามตัวอย่างข้างต้น  
- ห้ามเพิ่มข้อความหรือคำอธิบายอื่นใด
"""


    response = call_llm(prompt)

    import json
    try:
        result = json.loads(response)
    except Exception:
        result = {
            "question_quality_score": 1,
            "answer_quality_score": 1,
            "overall_qa_score": 1.0,
            "pass_fail_flag": "fail"
        }
    return {**qa_pair, **result}
