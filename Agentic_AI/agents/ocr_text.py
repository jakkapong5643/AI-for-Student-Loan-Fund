import pandas as pd

def load_ocr_text(input_path: str):

    df = pd.read_csv(input_path)
    records = df[['filename', 'text_ocr']].to_dict(orient='records')
    return records
