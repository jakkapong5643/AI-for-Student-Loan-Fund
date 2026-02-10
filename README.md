# AI for Student Loan Fund (AI สำหรับกองทุนเงินให้กู้ยืมเพื่อการศึกษา - กยศ.)

โปรเจกต์นี้มีวัตถุประสงค์เพื่อพัฒนาขั้นตอนการทำงานอัตโนมัติ (Automated Workflow) สำหรับการจัดการเอกสารของ กยศ. โดยใช้เทคโนโลยี Agentic AI และ OCR เพื่อเปลี่ยนเอกสารให้อยู่ในรูปแบบ Dataset สำหรับการตอบคำถาม (QA Pair Generation) อย่างมีประสิทธิภาพ

##  Key Features

- **OCR Processing**: แปลงไฟล์ PDF และรูปภาพเอกสารให้เป็นข้อความ (Text) โดยใช้โมเดล OCR ที่รองรับภาษาไทย
- **Agentic Workflow**: ใช้ **LangGraph** ในการควบคุมลำดับการทำงานของ AI Agents หลายตัวร่วมกัน (Multi-Agent System)
- **Text Cleaning & Quality Check**: ระบบทำความสะอาดข้อความและตรวจสอบคุณภาพความถูกต้องของข้อมูลก่อนนำไปใช้งาน
- **Automated QA Generation**: สร้างคู่คำถาม-คำตอบ (QA Pairs) จากเนื้อหาเอกสารโดยอัตโนมัติด้วย LLM (OpenAI)
- **Feedback Loop**: มีระบบตรวจสอบความถูกต้องและวนลูปเพื่อปรับปรุงคุณภาพ (Feedback Round) หากข้อมูลที่ได้ยังไม่ผ่านเกณฑ์

## Project Structure

```text
AI-for-Student-Loan-Fund/
├── Agentic_AI/             # ส่วนหลักของระบบ Agentic Workflow
│   ├── agents/             # รวม AI Agents ต่างๆ (OCR, Cleaner, Generator, Quality Checker)
│   ├── Manager/            # ส่วนควบคุม Pipeline และ Graph Logic
│   ├── utils/              # ฟังก์ชันช่วยเหลือและคำสั่งเสริม
│   ├── main.py             # จุดเริ่มต้นการรันระบบ workflow
│   └── requirements.txt    # รายการ library ที่จำเป็น
├── OCR/                    # สคริปต์และ Notebook สำหรับทดลองการทำ OCR (CID, PDF)
├── LLM.py                  # ส่วนเชื่อมต่อกับ Large Language Model
└── SimpleDataset.xlsx      # ชุดข้อมูลตัวอย่างสำหรับการทดสอบ
```

## Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/your-repo/AI-for-Student-Loan-Fund.git
   cd AI-for-Student-Loan-Fund
   ```

2. **ติดตั้ง Library ที่จำเป็น:**
   ```bash
   cd Agentic_AI
   pip install -r requirements.txt
   ```

3. **ตั้งค่า Environment Variables:**
   - ตรวจสอบการตั้งค่า API Key สำหรับ OpenAI ในระบบของคุณ

## Usage

สำหรับการรัน Workflow หลักของ AI Agents:
```bash
python Agentic_AI/main.py
```

ระบบจะอ่านข้อมูลจากไฟล์ Input ที่กำหนดไว้ใน `main.py` และเริ่มกระบวนการสกัดข้อมูล ทำความสะอาด และสร้าง QA Dataset ออกมาในโฟลเดอร์ Output

## Tech Stack

- **Language**: Python
- **Orchestration**: LangGraph
- **LLM**: OpenAI GPT Models
- **Thai NLP**: PyThaiNLP
- **OCR**: Python-based OCR libraries
- **Data Handling**: Pandas
