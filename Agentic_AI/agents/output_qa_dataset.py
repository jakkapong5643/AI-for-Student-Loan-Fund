
import pandas as pd
from datetime import datetime
import json
import os

def save_dataset(qa_pairs: list, output_path: str, cleaned_text: str = ""):
    file_exists = os.path.isfile(output_path)

    for idx, qa in enumerate(qa_pairs):
        record = {
            "question_text": qa.get("question_text"),
            "answer_text": qa.get("answer_text"),
            "answer_type": qa.get("answer_type"),
            "difficulty_level": qa.get("difficulty_level"),
            "question_quality_score": qa.get("question_quality_score"),
            "answer_quality_score": qa.get("answer_quality_score"),
            "overall_qa_score": qa.get("overall_qa_score"),
            "pass_fail_flag": qa.get("pass_fail_flag"),
            "feedback_round": qa.get("feedback_round", 0),
            "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cleaned_text": cleaned_text,
            "metadata": json.dumps({
                "agent": "AgenticQA_v1",
                "confidence": qa.get("overall_qa_score", 0),
            }),
            "source_language": "th"
        }

        df = pd.DataFrame([record])

        df.to_csv(output_path, mode='a', header=not file_exists, index=False)
        file_exists = True  

        print(f"Saved record {idx + 1} to {output_path}")
