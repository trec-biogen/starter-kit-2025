import torch
from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from pyserini.search.lucene import LuceneSearcher
from util import save_json, load_json
import re, json, random
import numpy as np
from typing import List, Dict
from tqdm.auto import tqdm



lucene_bm25_searcher = LuceneSearcher('../data/indexes/pubmed_baseline_collection_jsonl')
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)

adapter_path = "../data/model"
llama_model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_path,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
)
llama_tokenizer = AutoTokenizer.from_pretrained(adapter_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
reranker_model.to(device)
llama_model.to(device)

def set_th_config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_th_config(2025)

def retrieve_top_docs_and_pmids(query, top_k=100):
    hits = lucene_bm25_searcher.search(query, k=top_k)
    docs_and_pmids = []
    for hit in hits:
        try:
            pmid = int(hit.docid)
            content = json.loads(hit.lucene_document.get('raw'))['contents']  
            docs_and_pmids.append((content, pmid))
        except (ValueError, AttributeError):
            continue
    return docs_and_pmids


def rerank_documents(query, docs_and_pmids, reranker_model, tokenizer, device="cpu"):
    from torch.nn.functional import sigmoid

    inputs = [(query, doc) for doc, _ in docs_and_pmids]
    encoded = tokenizer.batch_encode_plus(
        inputs,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = reranker_model(**encoded).logits.squeeze(-1)

    scores = sigmoid(outputs) if outputs.ndim == 1 else outputs

    scored_docs = []
    for (doc, pmid), score in zip(docs_and_pmids, scores.cpu().numpy()):
        scored_docs.append((doc, pmid, float(score)))

    scored_docs.sort(key=lambda x: x[2], reverse=True)
    return scored_docs


def format_prompt_for_llama(query, top_docs):
    prompt = "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents."

    prompt += f"\nQuestion: {query}"
    for i, (doc, pmid, _) in enumerate(top_docs):
        citation_n=i+1
        prompt += f"[{citation_n}]:({doc})\n"
    prompt += f"\n\nAnswer:"
    return prompt

def generate_answer(query: str, reranker_model, reranker_tokenizer, llama_model, llama_tokenizer, device="cpu"):

    docs_and_pmids = retrieve_top_docs_and_pmids(query, top_k=100)
    reranked_docs = rerank_documents(query, docs_and_pmids, reranker_model, reranker_tokenizer, device)
    top_10_docs = reranked_docs[:10]
    prompt = format_prompt_for_llama(query, top_10_docs)
    inputs = llama_tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = llama_model.generate(
            **inputs,
            max_new_tokens=350,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    answer = llama_tokenizer.decode(outputs[0], skip_special_tokens=True).split('Answer:')[1]

    return answer, top_10_docs
def parse_response_with_citations(answer_text, citation_map):

    answer_text = re.sub(r'[–—]', '-', answer_text)

    citation_pattern = re.compile(r'\[(\s*\d+(?:\s*[-,]\s*\d+)*\s*)\]')

    citation_matches = list(citation_pattern.finditer(answer_text))

    sentence_spans = list(re.finditer(r'.+?[.!?](?=\s|$)', answer_text))

    responses = []

    for span in sentence_spans:
        sentence = span.group().strip()
        sentence_start, sentence_end = span.start(), span.end()

        pmids = set()

        for match in citation_matches:
            if sentence_start <= match.start() < sentence_end:
                citation_text = match.group(1)
                parts = re.split(r'[,]', citation_text)
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        # Handle ranges like 1-3
                        range_parts = part.split('-')
                        if len(range_parts) == 2 and range_parts[0].isdigit() and range_parts[1].isdigit():
                            start, end = int(range_parts[0]), int(range_parts[1])
                            for i in range(start, end + 1):
                                if 1 <= i <= len(citation_map):
                                    pmids.add(str(citation_map[i]))
                    elif part.isdigit():
                        i = int(part)
                        if 1 <= i <= len(citation_map):
                            pmids.add(str(citation_map[i]))

        cleaned_sentence = citation_pattern.sub('', sentence).strip()
        if cleaned_sentence:
            responses.append({
                "text": cleaned_sentence,
                "citations": sorted(pmids)
            })

    return responses
def generate_all_answers(
    input_json_path: str,
    output_json_path: str,
    team_id: str,
    run_id: str,
    reranker_model,
    reranker_tokenizer,
    llama_model,
    llama_tokenizer,
    device="cpu"
):
    data = load_json(input_json_path)
    results = []

    for item in tqdm(data):
        topic_id = str(item['id'])
        question = item['question']
        print(f"Processing Question ID {topic_id}...")

        answer_text, top_10_docs = generate_answer(
            question,
            reranker_model,
            reranker_tokenizer,
            llama_model,
            llama_tokenizer,
            device
        )
        citation_maps={}
        for i, (doc, pmid, _) in enumerate(top_10_docs):
            citation_maps[i+1]=pmid
        responses = parse_response_with_citations(answer_text, citation_maps)

        results.append({
            "metadata": {
                "team_id": team_id,
                "run_id": run_id,
                "topic_id": topic_id
            },
            "responses": responses
        })

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_json_path}")


generate_all_answers(
    input_json_path='../data/task_b.json',
    output_json_path='../data/task_b_output.json',
    team_id="organizers",
    run_id="organizers-run-example",
    reranker_model=reranker_model,
    reranker_tokenizer=reranker_tokenizer,
    llama_model=llama_model,
    llama_tokenizer=llama_tokenizer,
    device=device
)