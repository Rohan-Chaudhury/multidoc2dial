import os

from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import Type, List, Dict, Union, Tuple
from models import BEIRDPR
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder


query_encoder = DPRQuestionEncoder.from_pretrained(
            "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
query_tokenizer = AutoTokenizer.from_pretrained(
    "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")

doc_encoder = DPRContextEncoder.from_pretrained(
    "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
doc_tokenizer = AutoTokenizer.from_pretrained(
    "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")

beir_model = BEIRDPR(query_encoder, doc_encoder,
                         query_tokenizer, doc_tokenizer)
model = DRES(beir_model, batch_size=128)

corpus, queries, qrels = GenericDataLoader(
    "../data/mdd_dpr/beir_format/").load(split="train")

# print (qrels)
retriever = EvaluateRetrieval(model, score_function="dot", k_values=[20]) # retriever retrieves topk +1 for some reason
results = retriever.retrieve(corpus, queries)

os.makedirs("../data/mdd_dpr/dpr_negatives_beir_format/", exist_ok=True)
os.makedirs("../data/mdd_dpr/dpr_negatives_beir_format/qrels", exist_ok=True)

with open("../data/mdd_dpr/beir_format/corpus.jsonl", 'r') as fi:
    with open("../data/mdd_dpr/dpr_negatives_beir_format/corpus.jsonl", 'w') as fo:
        for line in fi:
            fo.write(line)

with open("../data/mdd_dpr/beir_format/queries.jsonl", 'r') as fi:
    with open("../data/mdd_dpr/dpr_negatives_beir_format/queries.jsonl", 'w') as fo:
        for line in fi:
            fo.write(line)

with open("../data/mdd_dpr/dpr_negatives_beir_format/qrels/train.tsv", 'w') as fo:
    fo.write('\t'.join(['query_id', 'doc_id', 'label']) + "\n")
    for query_id in results.keys():
        for doc_id, rel in qrels[query_id].items():
            if rel == 1:
                fo.write('\t'.join([query_id, doc_id, str(rel)]) + '\n')
        for doc_id, score in sorted(results[query_id].items(), key=lambda x: x[1], reverse=True):
            if qrels[query_id].get(doc_id, 0) == 0: # write only irrelevant results as negatives
                fo.write('\t'.join([query_id, doc_id, str(rel)]) + '\n')