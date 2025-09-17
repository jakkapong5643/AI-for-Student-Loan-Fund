import logging
from Manager.pipeline import create_workflow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logging.getLogger("AFC").setLevel(logging.ERROR)

def main():
    input_path = r"C:\Final Project\Code\Code\Agentic_AI\data\Input\test.csv"
    output_dir = r"C:\Final Project\Code\Code\Agentic_AI\data\output2"

    logging.info("Start workflow")
    graph = create_workflow(input_path, output_dir)

    init_state = {
        "messages": [],
        "index": 0,
        "filename": "",
        "text_ocr": "",
        "cleaned_text": "",
        "quality_score": 0,
        "quality_pass": False,
        "qa_plan": {},
        "qa_pairs": [],
        "qa_evaluation": [],
        "feedback_round": 0,
    }

    state = init_state
    while True:
        state = graph.invoke(state, config={"recursion_limit": 10_000})
        if not state:
            logging.info("completed")
            break

        logging.info(f"Current index: {state.get('index', '?')} | Filename: {state.get('filename', '')} | Feedback round: {state.get('feedback_round', 0)}")

if __name__ == "__main__":
    main()
