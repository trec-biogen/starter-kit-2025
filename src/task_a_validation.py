import os
import json
import sys
import glob
import logging
from colorama import Fore, Style, init

init(autoreset=True)

REQUIRED_QA_IDS = {f"{i}" for i in range(116, 116 + 40)}
ERROR_RETURN_CODE=255
def setup_logger(log_file='run.errlog', log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

def load_json_or_jsonl(file_path):
    if not os.path.isfile(file_path):
        logging.error(Fore.RED + f"Error: '{file_path}' is not a valid file.")
        sys.exit(ERROR_RETURN_CODE)
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
        f.seek(0)

        is_probably_jsonl = (
                first_line.startswith("{") and second_line.startswith("{")
        )

        if is_probably_jsonl:
            result = []
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(Fore.RED + f"Invalid JSONL format: {e}")
                    sys.exit(ERROR_RETURN_CODE)

            return result
        else:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                logging.error(Fore.RED + f"Invalid JSON format: {e}")
                sys.exit(ERROR_RETURN_CODE)


def validate_run_file(filepath):

    data = load_json_or_jsonl(filepath)

    if not isinstance(data, list):
        logging.error(Fore.RED + "Error: Expected a list of JSON items.")
        sys.exit(ERROR_RETURN_CODE)

    seen_qa_ids = set()

    for idx, item in enumerate(data):
        path_prefix = f"[Item {idx +1 }]"
        metadata = item.get('metadata', {})

        qa_id = metadata.get('qa_id')
        if not qa_id:
            logging.error(Fore.RED + f"{path_prefix} Missing 'qa_id' in metadata.")
            sys.exit(ERROR_RETURN_CODE)

        if qa_id not in REQUIRED_QA_IDS:
            logging.error(Fore.RED + f"{path_prefix} Unexpected qa_id: {qa_id}")
            sys.exit(ERROR_RETURN_CODE)

        seen_qa_ids.add(qa_id)


        answers = item.get('answer', [])
        if not isinstance(answers, list):
            logging.error(Fore.RED + f"{path_prefix} 'answer' must be a list.")
            sys.exit(ERROR_RETURN_CODE)

        for a_idx, ans in enumerate(answers):
            answer_prefix = f"{path_prefix} [Answer {a_idx+1}]"
            for field in ["text", "supported_citations", "contradicted_citations"]:
                if field not in ans:
                    logging.error(Fore.RED + f"{answer_prefix} Missing field '{field}'.")
                    sys.exit(ERROR_RETURN_CODE)

            existing = ans.get("existing_supported_citations", [])
            supported = ans.get("supported_citations", [])
            contradicted = ans.get("contradicted_citations", [])

            if existing is None:
                existing = []

            overlap_supported = set(supported) & set(existing)
            overlap_contradicted = set(contradicted) & set(existing)

            if overlap_supported:
                logging.error(Fore.RED + f"{answer_prefix} Overlap in supported and existing citations: {overlap_supported}")
                sys.exit(ERROR_RETURN_CODE)


    missing_ids = REQUIRED_QA_IDS - seen_qa_ids
    if missing_ids:
        logging.error(Fore.RED + f"\nMissing {len(missing_ids)} qa_ids: {sorted(missing_ids)}")
        sys.exit(ERROR_RETURN_CODE)

    else:
        logging.info(Fore.GREEN + "Success: Run file validated successfully.")
        sys.exit(0)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{Fore.YELLOW}Usage: python task_a_validation.py <path_to_run_file>{Style.RESET_ALL}")
    else:
        logging = setup_logger(log_file=sys.argv[1]+'.errlog')
        validate_run_file(sys.argv[1])

