from transformers import AutoModelForSeq2SeqLM
from transformers import  AutoTokenizer

from pyserini.search.lucene import LuceneSearcher
import json
from tqdm.auto import tqdm
from util import save_json, load_json
import torch
import numpy as np
import random
def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_th_config(2025)
lucene_bm25_searcher = LuceneSearcher('../data/indexes/pubmed_baseline_collection_jsonl')

tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")
model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def check_contradiction(sentences1: str, sentence2: str) -> str:
    prompts = [f"mednli: sentence1: {s1} sentence2: {sentence2}" for s1 in sentences1]

    encodings = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=8,
            early_stopping=True
        )

    results = [
        tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True).lower()
        for output in outputs
    ]

    return results

def retrieve_top_pmids(query, top_k=20):
    hits = lucene_bm25_searcher.search(query, k=top_k)
    pmids = []
    for hit in hits:
        try:
            pmid = int(hit.docid)
            pmids.append(pmid)
        except ValueError:
            continue
    return pmids

def retrieve_top_pmids_with_contents(query, top_k=20):
    hits = lucene_bm25_searcher.search(query, k=top_k)
    pmids = []
    for hit in hits:
        try:
            pmid = int(hit.docid)
            pmids.append((pmid, hit.lucene_document.get('raw')))
        except ValueError:
            continue
    return pmids

def enrich_with_supported_and_contradicted(input_json_path, output_json_path):
    data = load_json(input_json_path)

    for item in tqdm(data):
        for answer in item.get("answer", []):
            original_query = answer["text"]
            existing_pmids = set(answer.get("existing_supported_citations") or [])

            # --- Supported Citations ---
            retrieved_pmids = retrieve_top_pmids(original_query, top_k=25)
            new_supported = [pmid for pmid in retrieved_pmids if pmid not in existing_pmids][:3]
            answer["supported_citations"] = new_supported

            # --- Contradicted Citations ---
            hits = retrieve_top_pmids_with_contents(original_query, top_k=500)
            docs =[json.loads(x[1])['contents'] for x in hits]
            pmids =[x[0] for x in hits]

            results = check_contradiction(docs, original_query)
            negative_docs = []
            assert len(results) == len(pmids)
            for result, pmid in zip(results, pmids):
                if result == 'contradiction':
                    negative_docs.append(pmid)

            new_contradicted = [
                pmid for pmid in negative_docs
                if pmid not in existing_pmids and pmid not in new_supported
            ][:3]
            answer["contradicted_citations"] = new_contradicted

    save_json(data, output_json_path)


enrich_with_supported_and_contradicted('../data/task_a.json', '../data/task_a_output.json')