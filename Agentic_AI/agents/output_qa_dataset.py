import pandas as pd
from datetime import datetime
import json

def save_dataset(qa_pairs: list, output_path: str):

    records = []
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
            "metadata": json.dumps({
                "agent": "AgenticQA_v1",
                "confidence": qa.get("overall_qa_score", 0),
            }),
            "source_language": "th"
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")
