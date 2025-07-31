import pandas as pd

def load_ocr_text(input_path: str):
    """
    โหลดข้อความ OCR ภาษาไทยจาก CSV (col: filename, text_ocr)
    คืนค่า list dict [{'filename':..., 'text_ocr':...}, ...]
    """
    df = pd.read_csv(input_path)
    records = df[['filename', 'text_ocr']].to_dict(orient='records')
    return records
