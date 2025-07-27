import os
import json
import sys
import glob
import logging
from colorama import Fore, Style, init

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format='%(message)s')

REQUIRED_QA_IDS = {f"{i}" for i in range(116, 116 + 40)}
def load_json_or_jsonl(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
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
                    raise ValueError(f"Invalid JSONL at line {i}: {str(e)}")
            return result
        else:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {str(e)}")

def validate_run_file(filepath):
    if not os.path.isfile(filepath):
        logging.error(Fore.RED + f"Error: '{filepath}' is not a valid file.")
        return

    try:
        data = load_json_or_jsonl(filepath)
    except json.JSONDecodeError as e:
        logging.error(Fore.RED + f"Invalid JSON format: {e}")
        return

    if not isinstance(data, list):
        logging.error(Fore.RED + "Error: Expected a list of JSON items.")
        return

    seen_qa_ids = set()

    for idx, item in enumerate(data):
        path_prefix = f"[Item {idx +1 }]"
        metadata = item.get('metadata', {})

        qa_id = metadata.get('qa_id')
        if not qa_id:
            logging.error(Fore.RED + f"{path_prefix} Missing 'qa_id' in metadata.")
            return

        if qa_id not in REQUIRED_QA_IDS:
            logging.error(Fore.YELLOW + f"{path_prefix} Unexpected qa_id: {qa_id}")
            return
        seen_qa_ids.add(qa_id)


        answers = item.get('answer', [])
        if not isinstance(answers, list):
            logging.error(Fore.RED + f"{path_prefix} 'answer' must be a list.")
            return

        for a_idx, ans in enumerate(answers):
            answer_prefix = f"{path_prefix} [Answer {a_idx}]"
            for field in ["text", "supported_citations", "contradicted_citations"]:
                if field not in ans:
                    logging.error(Fore.RED + f"{answer_prefix} Missing field '{field}'.")
                    return
            existing = ans.get("existing_supported_citations", [])
            supported = ans.get("supported_citations", [])
            contradicted = ans.get("contradicted_citations", [])

            if existing is None:
                existing = []

            overlap_supported = set(supported) & set(existing)
            overlap_contradicted = set(contradicted) & set(existing)

            if overlap_supported:
                logging.error(Fore.RED + f"{answer_prefix} Overlap in supported and existing citations: {overlap_supported}")
                return
            if overlap_contradicted:
                logging.error(Fore.RED + f"{answer_prefix} Overlap in contradicted and existing citations: {overlap_contradicted}")
                return

    missing_ids = REQUIRED_QA_IDS - seen_qa_ids
    if missing_ids:
        logging.error(Fore.YELLOW + f"\nMissing {len(missing_ids)} qa_ids: {sorted(missing_ids)}")
        return
    else:
        logging.info(Fore.GREEN + "\nSuccess: Run file validated successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{Fore.YELLOW}Usage: python task_b_validation.py <path_to_run_file>{Style.RESET_ALL}")
    else:
        validate_run_file(sys.argv[1])

