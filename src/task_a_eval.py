import glob, os, re, string
from collections import defaultdict
import json
import csv
import pandas as pd

import argparse
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def load_json(path):
    with open(path, 'r') as rfile:
        return json.load(rfile)

def load_json_or_jsonl(file_path):
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
                    print(f"Invalid JSONL format: {e}")

            return result
        else:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON format: {e}")


def normalize_sents(text):

    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\.{2,}", "...", text)
    punctuation = string.punctuation.replace("-", "")
    text = re.sub(rf"\s+([{re.escape(punctuation)}])", r"\1", text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(rf"([{re.escape(string.punctuation)}])(?!\s|$)", r"\1 ", text)

    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"\{\s+", "{", text)
    text = re.sub(r"\s+\}", "}", text)

    text = re.sub(r'([\'"]) +', r"\1", text)
    text = re.sub(r' +([\'"])', r"\1", text)

    text = " ".join(text.split())

    return text




def extract_pmid(citation):
    for line in citation.splitlines():
        line = line.strip().lower()
        if line.startswith("pmid:"):
            return line.split("pmid:")[1].strip()
    return None


def normalize_judgement(j):
    j = j.lower().strip()
    if j in ["supports", "supported"]:
        return "supports"
    if j in ["contradicted", "contradicts"]:
        return "contradicts"
    if j in ["partial-supported", "partially supports", "partial support",
             "partially supported", "partially-support"]:
        return "partial_supports"
    if j == "neutral":
        return "neutral"
    if j in ["not relevant", "not_relevant"]:
        return "not_relevant"
    print("Using fallback judgement mapping")
    return j


def prec(tp, fp):
    return tp / (tp + fp) if tp + fp > 0 else 0.0

def rec(tp, fn):
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def evaluate_runs(path_to_baseline_submission, path_to_judgement):
    question2pmid_data = {}
    question2ext_pmid_data = {}
    questionid2question = {}

    loaded_data = load_json_or_jsonl(path_to_baseline_submission)
    for data_item in loaded_data:
        qa_id = data_item['metadata']['qa_id']
        if qa_id not in questionid2question:
            questionid2question[qa_id] = data_item['metadata']['question']
        if qa_id not in question2ext_pmid_data:
            question2ext_pmid_data[qa_id] = {}
        for answer in data_item['answer']:
            answer_text = normalize_sents(answer['text'])
            supported_citations = answer['existing_supported_citations']
            if supported_citations is None:
                supported_citations = []

            supported_citations = [str(x) for x in supported_citations]

            if answer_text not in question2ext_pmid_data[qa_id]:
                question2ext_pmid_data[qa_id][answer_text] = {'supported_citations': supported_citations}
            else:
                question2ext_pmid_data[qa_id][answer_text]['supported_citations'].extend(supported_citations)


    run_name = path_to_baseline_submission.split('/')[-1]
    loaded_data = load_json_or_jsonl(path_to_baseline_submission)
    for data_item in loaded_data:
        qa_id = data_item['metadata']['qa_id']
        assert len(data_item['answer']) == len(question2ext_pmid_data[qa_id])
        if qa_id not in question2pmid_data:
            question2pmid_data[qa_id] = {}
        if run_name not in question2pmid_data[qa_id]:
            question2pmid_data[qa_id][run_name] = {}
        for answer in data_item['answer']:
            answer_text = normalize_sents(answer['text'])
            supported_citations = answer['supported_citations']
            supported_citations = [str(x) for x in supported_citations]

            if 'contradicted_citations' in answer:
                contradicted_citations = answer['contradicted_citations']
            elif 'contradicting_citations' in answer:
                contradicted_citations = answer['contradicting_citations']
            else:
                print("Error...")
                exit(-1)
            contradicted_citations = [str(x) for x in contradicted_citations]
            ext_supp_citations = question2ext_pmid_data[qa_id][answer_text]['supported_citations']

            filtered_supp_citations = []
            for supp_citations in supported_citations:
                if supp_citations not in ext_supp_citations:
                    filtered_supp_citations.append(supp_citations)

            if answer_text not in question2pmid_data[qa_id][run_name]:
                question2pmid_data[qa_id][run_name][answer_text] = {
                    'supported_citations': filtered_supp_citations[:3],
                    'contradicted_citations': contradicted_citations[:3]}
            else:
                question2pmid_data[qa_id][run_name][answer_text]['supported_citations'].extend(
                    filtered_supp_citations[:3])
                question2pmid_data[qa_id][run_name][answer_text]['contradicted_citations'].extend(
                    contradicted_citations[:3])
        assert len(question2pmid_data[qa_id][run_name]) == len(question2ext_pmid_data[qa_id])

    csv_judgments = defaultdict(dict)
    qids_in_csv = set()

    with open(path_to_judgement, "r") as f:
        reader = csv.DictReader(f, delimiter=",")
        for row in reader:
            qid = row["qid"].strip()
            sentence = row["answer_sentence"].strip()
            pmid = extract_pmid(row["citation"])
            if not pmid:
                continue

            qids_in_csv.add(qid)
            csv_judgments[qid][(sentence, pmid)] = normalize_judgement(row["judgement"])

    rows = []
    missed_qids = []

    for qid, runs in question2pmid_data.items():

        if qid not in qids_in_csv:
            missed_qids.append(qid)
            continue

        for run_name, sent_dict in runs.items():
            for sentence, pred in sent_dict.items():
                supported_pred = set(pred.get("supported_citations", []))
                contrad_pred = set(pred.get("contradicted_citations", []))
                if len(supported_pred) == 0:
                    print("ll")

                if len(contrad_pred) == 0:
                    print("ll")
                csv_pmids = {
                    pmid for (sent, pmid) in csv_judgments[qid] if sent == sentence
                }

                if len(supported_pred) > 0 and len(supported_pred & csv_pmids) == 0:
                    print(f"[WARNING] qid={qid} sentence='{sentence}' → no supported PMIDs in CSV")
                if len(contrad_pred) > 0 and len(contrad_pred & csv_pmids) == 0:
                    print(f"[WARNING] qid={qid} sentence='{sentence}' → no contradicted PMIDs in CSV")

                TP_sup = FP_sup = 0
                TP_con = FP_con = 0

                for pmid in supported_pred:
                    if (sentence, pmid) not in csv_judgments[qid]:
                        continue
                    j = csv_judgments[qid][(sentence, pmid)]
                    if j == "supports":
                        TP_sup += 1
                    else:
                        FP_sup += 1

                for pmid in contrad_pred:
                    if (sentence, pmid) not in csv_judgments[qid]:
                        continue
                    j = csv_judgments[qid][(sentence, pmid)]
                    if j == "contradicts":
                        TP_con += 1
                    else:
                        FP_con += 1

                strict_support_prec = prec(TP_sup, FP_sup)
                strict_contra_prec = prec(TP_con, FP_con)

                judged_supports = {
                    pmid for (sent, pmid), j in csv_judgments[qid].items()
                    if sent == sentence and j == "supports"
                }
                judged_contrads = {
                    pmid for (sent, pmid), j in csv_judgments[qid].items()
                    if sent == sentence and j == "contradicts"
                }

                soft_recall_sup = 1 if supported_pred & judged_supports else 0
                soft_recall_con = 1 if contrad_pred & judged_contrads else 0

                TP_sup_r = FP_sup_r = 0
                for pmid in supported_pred:
                    if (sentence, pmid) not in csv_judgments[qid]:
                        continue
                    j = csv_judgments[qid][(sentence, pmid)]
                    if j in ["supports", "partial_supports"]:
                        TP_sup_r += 1
                    else:
                        FP_sup_r += 1

                relaxed_support_prec = prec(TP_sup_r, FP_sup_r)
                relaxed_contra_prec = strict_contra_prec  # contradicted same

                judged_relaxed_supports = {
                    pmid for (sent, pmid), j in csv_judgments[qid].items()
                    if sent == sentence and j in ["supports", "partial_supports"]
                }
                soft_recall_sup_r = 1 if supported_pred & judged_relaxed_supports else 0
                soft_recall_con_r = soft_recall_con  # contradicted same

                row = {
                    "qid": qid,
                    "run": run_name,
                    "sentence": sentence,

                    "strict_support_precision": strict_support_prec,
                    "strict_support_soft_recall": soft_recall_sup,
                    "strict_contradict_precision": strict_contra_prec,
                    "strict_contradict_soft_recall": soft_recall_con,

                    "relaxed_support_precision": relaxed_support_prec,
                    "relaxed_support_soft_recall": soft_recall_sup_r,
                    "relaxed_contradict_precision": relaxed_contra_prec,
                    "relaxed_contradict_soft_recall": soft_recall_con_r
                }
                rows.append(row)


    df = pd.DataFrame(rows)
    metric_cols = [
        'strict_support_precision', 'strict_support_soft_recall',
        'strict_contradict_precision', 'strict_contradict_soft_recall',
        'relaxed_support_precision', 'relaxed_support_soft_recall',
        'relaxed_contradict_precision', 'relaxed_contradict_soft_recall'
    ]


    per_qid_avg = df.groupby(['run', 'qid'])[metric_cols].mean().mul(100).round(2).reset_index()
    per_qid_avg['sentence'] = 'AVERAGE_OVER_SENTENCES'

    overall_avg = per_qid_avg.groupby(['run'])[metric_cols].mean().round(2).reset_index()
    overall_avg['qid'] = 'AVERAGE_OVER_QIDS'
    overall_avg['sentence'] = 'AVERAGE_OVER_SENTENCES'
    print(f"Missing qids: {len(missed_qids)}")
    print(overall_avg)
def precision_recall_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1
def normalize_label(label):
    if label == "Supports" or label == "Support":
        return "SUPPORT"
    if label == "Contradict" or label == "Contradicts":
        return "CONTRADICT"
    if label == "Not relevant" or label == "Neutral":
        return "OTHER"
    else:
        print(f"Undefined label: {label}")
    return "OTHER"
def evaluate_item(item):
    gold = {}

    for pmid, label in item["supported_citations_labels"].items():
        gold[str(pmid)] = normalize_label(label)

    for pmid, label in item["contradicted_citations_labels"].items():
        gold[str(pmid)] = normalize_label(label)

    pred_supported = set([str(x) for x in item["supported_citations"]])
    pred_contradicted = set([str(x) for x in item["contradicted_citations"]])

    pred = {}
    for pmid in gold:
        if pmid in pred_supported:
            pred[pmid] = "SUPPORT"
        elif pmid in pred_contradicted:
            pred[pmid] = "CONTRADICT"
        else:
            pred[pmid] = "OTHER"

    tp_s = fp_s = fn_s = 0
    tp_c = fp_c = fn_c = 0

    for pmid in gold:
        g = gold[pmid]
        p = pred[pmid]

        if g == "SUPPORT" and p == "SUPPORT":
            tp_s += 1
        elif g != "SUPPORT" and p == "SUPPORT":
            fp_s += 1
        elif g == "SUPPORT" and p != "SUPPORT":
            fn_s += 1

        if g == "CONTRADICT" and p == "CONTRADICT":
            tp_c += 1
        elif g != "CONTRADICT" and p == "CONTRADICT":
            fp_c += 1
        elif g == "CONTRADICT" and p != "CONTRADICT":
            fn_c += 1

    p_s, r_s, f1_s = precision_recall_f1(tp_s, fp_s, fn_s)
    p_c, r_c, f1_c = precision_recall_f1(tp_c, fp_c, fn_c)

    return {
        "support": {
            "tp": tp_s, "fp": fp_s, "fn": fn_s,
            "precision": p_s, "recall": r_s, "f1": f1_s,
        },
        "contradict": {
            "tp": tp_c, "fp": fp_c, "fn": fn_c,
            "precision": p_c, "recall": r_c, "f1": f1_c,
        },
    }


def evaluate_task_a_using_automatic_evaluation(path_to_baseline_submission):
    res_list=[]
    run_name = path_to_baseline_submission.split("/")[-1].replace("_labels.json", "")
    team_name = path_to_baseline_submission.split("/")[-2]
    meta_data = {"team_id": team_name, 'run_id': run_name}
    data_list = load_json(path_to_baseline_submission)
    all_results = []

    for data in data_list:
        for item in data["answer"]:
            all_results.append(evaluate_item(item))

    def macro_avg(cls, metric):
        return round((sum(r[cls][metric] for r in all_results) / len(all_results))*100, 2)

    final_metrics = {
        "supported_precision": macro_avg("support", "precision"),
        "supported_recall": macro_avg("support", "recall"),
        "supported_f1": macro_avg("support", "f1"),
        "contradicted_precision": macro_avg("contradict", "precision"),
        "contradicted_recall": macro_avg("contradict", "recall"),
        "contradicted_f1": macro_avg("contradict", "f1"),
    }
    meta_data.update(final_metrics)
    res_list.append(meta_data)
    print(res_list)



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Task-A submissions and automatic evaluation labels."
    )

    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to baseline submission JSON file"
    )

    parser.add_argument(
        "--judgement",
        type=str,
        required=True,
        help="Path to judgement CSV file"
    )

    parser.add_argument(
        "--labels",
        type=str,
        required=True,
        help="Path to automatic evaluation labels JSON file"
    )

    args = parser.parse_args()

    evaluate_runs(
        path_to_baseline_submission=args.submission,
        path_to_judgement=args.judgement
    )

    evaluate_task_a_using_automatic_evaluation(
        path_to_baseline_submission=args.labels
    )

if __name__ == "__main__":
    main()