import json
import sys
import os
import logging
import glob
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)


class ColorFormatter(logging.Formatter):
    def format(self, record):
        level_color = {
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'DEBUG': Fore.CYAN
        }.get(record.levelname, '')

        message = super().format(record)
        return f"{level_color}{message}{Style.RESET_ALL}"


handler = logging.StreamHandler()
formatter = ColorFormatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [handler]


def validate_entry(entry, index):
    errors = []
    warnings =[]
    path_prefix = f"[Item {index}]"

    if not isinstance(entry, dict):
        errors.append(f"{path_prefix} is not a dictionary.")
        return errors

    required_keys = {"metadata", "responses"}
    missing_keys = required_keys - entry.keys()
    extra_keys = entry.keys() - required_keys

    if missing_keys:
        errors.append(f"{path_prefix} Missing required fields: {', '.join(missing_keys)}.")
    if extra_keys:
        warnings.append(f"{path_prefix} Contains extra fields: {', '.join(extra_keys)}.")

    # Metadata checks
    metadata = entry.get("metadata")
    if not isinstance(metadata, dict):
        errors.append(f"{path_prefix} 'metadata' must be a dictionary.")
    else:
        for key in ["team_id", "run_id"]:
            if key not in metadata:
                errors.append(f"{path_prefix} 'metadata' is missing key: '{key}'.")
            elif not isinstance(metadata[key], str):
                actual_type = type(metadata[key]).__name__
                errors.append(f"{path_prefix} 'metadata.{key}' must be a string, got {actual_type}.")

        for key in ["topic_id"]:
            if key not in metadata:
                errors.append(f"{path_prefix} 'metadata' is missing key: '{key}'.")
            elif not isinstance(metadata[key], str) and not isinstance(metadata[key], int):
                actual_type = type(metadata[key]).__name__
                errors.append(f"{path_prefix} 'metadata.{key}' must be a string/int, got {actual_type}.")

    # Responses checks
    responses = entry.get("responses")
    if not isinstance(responses, list):
        errors.append(f"{path_prefix} 'responses' must be a list.")
    else:
        for i, response in enumerate(responses):
            resp_path = f"{path_prefix} > responses[{i+1}]"
            if not isinstance(response, dict):
                errors.append(f"{resp_path} must be a dictionary.")
                continue

            if "text" not in response:
                errors.append(f"{resp_path} Missing field: 'text'.")
            elif not isinstance(response["text"], str):
                actual_type = type(response["text"]).__name__
                errors.append(f"{resp_path} 'text' must be a string, got {actual_type}.")

            if "citations" not in response:
                errors.append(f"{resp_path} Missing field: 'citations'.")
            elif not isinstance(response["citations"], list) and not isinstance(response["citations"], dict):
                actual_type = type(response["citations"]).__name__
                errors.append(f"{resp_path} 'citations' must be a dict/list, got {actual_type}.")
            else:
                if isinstance(response["citations"], list):
                    for j, citation in enumerate(response["citations"]):
                        if not isinstance(citation, int) and not isinstance(citation, str):
                            actual_type = type(citation).__name__
                            errors.append(f"{resp_path} > citations[{j+1}] must be string/int, got {actual_type}.")

                elif isinstance(response["citations"], dict):
                    citation_list=response["citations"].keys()
                    for j, citation in enumerate(citation_list):
                        if not isinstance(citation, int) and not isinstance(citation, str):
                            actual_type = type(citation).__name__
                            errors.append(f"{resp_path} > citations[{j+1}] must be string/int, got {actual_type}.")

    return errors, warnings


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



def validate_run_file(filename, topic_size=30, topic_id_start=181):
    topic_ids = list(range(topic_id_start, topic_id_start + topic_size))
    try:
        data = load_json_or_jsonl(filename)

        if not isinstance(data, list):
            logger.error("Top-level JSON must be a list.")
            return

        if len(data) != topic_size:
            logger.error("Mismatch in number of topics.")
            logger.error(f"Expected topics: {topic_size}, Found: {len(data)}")
            return

        all_errors = []
        all_warnings = []

        found_topic_ids = set()
        invalid_topic_ids = []

        for idx, entry in enumerate(data):
            metadata = entry.get("metadata", {})
            topic_id_raw = metadata.get("topic_id")

            if isinstance(topic_id_raw, str) and topic_id_raw.isdigit():
                topic_id = int(topic_id_raw)
            else:
                topic_id = topic_id_raw

            if topic_id not in topic_ids:
                invalid_topic_ids.append((idx, topic_id))

            found_topic_ids.add(topic_id)
            entry_errors, entry_warnings = validate_entry(entry, idx)
            all_errors.extend(entry_errors)
            all_warnings.extend(entry_warnings)
        missing_ids = set(topic_ids) - found_topic_ids

        if invalid_topic_ids:
            for idx, bad_id in invalid_topic_ids:
                logger.error(f"[Entry {idx+1}] Invalid or missing topic_id: {bad_id}")
                return
        if missing_ids:
            logger.error(f"Missing topic_ids: {sorted(missing_ids)}")
            return
        if all_errors:
            logger.error("Validation failed with the following issues:")
            for err in all_errors:
                logger.error(err)
            return
        if all_warnings:
            logger.warning("Validation warnings:")
            for warns in all_warnings:
                logger.warning(warns)

        elif not missing_ids and not invalid_topic_ids:
            logger.info("Success: Run file validated successfully.")

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"{Fore.YELLOW}Usage: python task_b_validation.py <path_to_run_file>{Style.RESET_ALL}")
    else:
        validate_run_file(sys.argv[1])

