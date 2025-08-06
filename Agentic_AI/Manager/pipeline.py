import logging
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta
import os

start_time = datetime.now()

from agents import (
    ocr_text,
    text_cleaner,
    text_quality_checker,
    question_planner,
    qa_generator,
    qa_evaluator,
    output_qa_dataset,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)

QUALITY_THRESHOLD = 3
MAX_FEEDBACK_ROUNDS = 3

class State(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    index: int
    filename: str
    text_ocr: str
    cleaned_text: str
    quality_score: int
    quality_pass: bool
    qa_plan: Dict[str, Any]
    qa_pairs: List[Dict[str, Any]]
    qa_evaluation: List[Dict[str, Any]]
    text_feedback_round: int
    qa_feedback_round: int
    quality_feedback: str
    qa_feedback: str

def create_workflow(input_path: str, output_dir: str):
    ocr_records = ocr_text.load_ocr_text(input_path)

    graph_builder = StateGraph(State)

    def node_load_ocr(state: State) -> dict:
        idx = state.get("index", 0)
        total = len(ocr_records)

        if idx < total:
            record = ocr_records[idx]
            logger.info(f"[{idx+1}/{total}] Loading OCR record: filename='{record['filename']}'")

            return {
                "filename": record["filename"],
                "text_ocr": record["text_ocr"],
                "text_feedback_round": 0,
                "qa_feedback_round": 0,
                "quality_feedback": "",
                "qa_feedback": "",
            }
        else:
            logger.info(f"All {total} records processed. Workflow is complete.")
            return {}

    def node_clean_text(state: State) -> dict:
        raw = state.get("text_ocr", "")
        feedback = state.get("quality_feedback", "")
        logger.info(f"Cleaning text filename={state.get('filename', 'unknown')}, text_feedback_round={state.get('text_feedback_round', 0)}")

        cleaned = text_cleaner.clean_text(raw, feedback_reason=feedback)

        logger.info(f"Completed cleaning filename={state.get('filename', 'unknown')}")
        return {"cleaned_text": cleaned}

    def node_quality_check(state: State) -> dict:
        cleaned_text = state.get("cleaned_text", "")
        score = text_quality_checker.evaluate_text_quality(cleaned_text)
        passed = score >= QUALITY_THRESHOLD

        logger.info(f"Quality check {state.get('filename', 'unknown')}: score={score}, pass={passed}")

        return {"quality_score": score, "quality_pass": passed}

    def node_feedback_text(state: State) -> dict:
        quality_pass = state.get("quality_pass", False)
        text_feedback_round = state.get("text_feedback_round", 0)

        if quality_pass:
            logger.info(f"Text quality Pass filename={state.get('filename', 'unknown')}, resetting feedback.")
            return {
                "quality_feedback": "",
                "text_feedback_round": 0,
            }

        if text_feedback_round >= MAX_FEEDBACK_ROUNDS:
            logger.error(f"Max text feedback rounds filename={state.get('filename', 'unknown')}, skipping cleaning.")
            return {"text_feedback_round": MAX_FEEDBACK_ROUNDS}

        feedback_reason = text_quality_checker.get_feedback_reason(state.get("cleaned_text", ""))
        logger.info(f"Feedback for text cleaning (round {text_feedback_round + 1}): {feedback_reason}")

        return {
            "text_feedback_round": text_feedback_round + 1,
            "text_ocr": state.get("cleaned_text", ""),
            "quality_feedback": feedback_reason,
        }

    def node_question_plan(state: State) -> dict:
        logger.info(f"Planning questions filename={state.get('filename', 'unknown')}")
        plan = question_planner.plan_questions(state.get("cleaned_text", ""))
        logger.info(f"Completed planning questions filename={state.get('filename', 'unknown')}")
        return {"qa_plan": plan}

    def node_generate_qa(state: State) -> dict:
        feedback = state.get("qa_feedback", "")
        qas = qa_generator.generate_qa(
            text=state.get("cleaned_text", ""),
            plan=state.get("qa_plan", {}),
            feedback_reason=feedback
        )

        logger.info(f"Generated {len(qas)} QA pairs for {state.get('filename', 'unknown')}")
        return {"qa_pairs": qas}

    def node_evaluate_qa(state: State) -> dict:
        logger.info(f"Evaluating QA pairs for filename={state.get('filename', 'unknown')}")
        evals = [qa_evaluator.evaluate(qa) for qa in state.get("qa_pairs", [])]
        logger.info(f"Completed evaluating {len(evals)} QA pairs for filename={state.get('filename', 'unknown')}")
        return {"qa_evaluation": evals}

    def node_feedback_qa(state: State) -> dict:
        qa_feedback_round = state.get("qa_feedback_round", 0)
        evaluations = state.get("qa_evaluation", [])
        passed = all(qa.get("pass_fail_flag") == "pass" for qa in evaluations)

        if passed:
            logger.info(f"All QA pairs passed evaluation for filename={state.get('filename', 'unknown')}, resetting QA feedback.")
            return {
                "qa_feedback": "",
                "qa_feedback_round": 0,
            }

        if qa_feedback_round >= MAX_FEEDBACK_ROUNDS:
            logger.error(f"Max QA feedback rounds reached for filename={state.get('filename', 'unknown')}, skipping regeneration.")
            return {"qa_feedback_round": MAX_FEEDBACK_ROUNDS}

        failed_feedbacks = [qa.get("feedback_reason", "") for qa in evaluations if qa.get("pass_fail_flag") != "pass"]
        combined_feedback = "\n".join(filter(None, failed_feedbacks))

        logger.info(f"Feedback for QA regeneration (round {qa_feedback_round + 1}): {combined_feedback}")

        return {
            "qa_feedback_round": qa_feedback_round + 1,
            "qa_feedback": combined_feedback,
        }

    def node_save_qa(state: State) -> dict:
        filename = state.get("filename", f"unknown_{state.get('index', 0)}")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"QA_{filename}.csv")
        output_qa_dataset.save_dataset(
            qa_pairs=state.get("qa_evaluation", []),
            output_path=save_path,
            cleaned_text=state.get("cleaned_text", "")
        )
        
        logger.info(f"Saved QA dataset for filename={filename} at {save_path}")

        i = state.get("index", 0) + 1
        N = len(ocr_records)
        logger.info(f"Progress: {i}/{N} records ({(i/N)*100:.2f}%)")

        elapsed = (datetime.now() - start_time).total_seconds() / 60 
        T_avg = elapsed / i if i > 0 else 1
        est_remaining = N - i
        est_finish = datetime.now() + timedelta(minutes=est_remaining * T_avg)
        logger.info(f"Estimated finish time: {est_finish.strftime('%Y-%m-%d %H:%M:%S')} (avg: {T_avg:.2f} min/record)")

        return {"index": i}

    graph_builder.add_node("Load_OCR", node_load_ocr)
    graph_builder.add_node("Clean_Text", node_clean_text)
    graph_builder.add_node("Quality_Check", node_quality_check)
    graph_builder.add_node("Feedback_Text", node_feedback_text)
    graph_builder.add_node("Question_Plan", node_question_plan)
    graph_builder.add_node("Generate_QA", node_generate_qa)
    graph_builder.add_node("Evaluate_QA", node_evaluate_qa)
    graph_builder.add_node("Feedback_QA", node_feedback_qa)
    graph_builder.add_node("Save_QA", node_save_qa)

    graph_builder.add_edge(START, "Load_OCR")
    graph_builder.add_edge("Load_OCR", "Clean_Text")
    graph_builder.add_edge("Clean_Text", "Quality_Check")
    graph_builder.add_edge("Quality_Check", "Feedback_Text")

    graph_builder.add_edge("Feedback_Text", "Clean_Text")
    graph_builder.add_edge("Feedback_Text", "Question_Plan")
    graph_builder.add_edge("Question_Plan", "Generate_QA")
    graph_builder.add_edge("Generate_QA", "Evaluate_QA")
    graph_builder.add_edge("Evaluate_QA", "Feedback_QA")

    graph_builder.add_edge("Feedback_QA", "Generate_QA")
    graph_builder.add_edge("Feedback_QA", "Save_QA")

    graph_builder.add_edge("Save_QA", "Load_OCR")  
    graph_builder.add_edge("Load_OCR", "Clean_Text")

    graph = graph_builder.compile()
    return graph