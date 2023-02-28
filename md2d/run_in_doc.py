import json
from beir import util
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import Type, List, Dict, Union, Tuple
from models import Splade, BEIRSpladeModel
from beir.retrieval.custom_metrics import mrr
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder

import argparse
import torch
import pdb


model_type_or_dir = "/home/adityasv/multidoc2dial/md2d/splade/training_with_sentence_transformers/output/distilsplade-ft-dpr/0_MLMTransformer"
model = Splade(model_type_or_dir)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
beir_splade = BEIRSpladeModel(model, tokenizer)

corpus, queries, qrels = GenericDataLoader(
    "/home/adityasv/multidoc2dial/retrieval_data").load(split="dev")

data = json.load(open('/home/adityasv/multidoc2dial/md2d/in-doc-negatives/analysis_in_doc.json', 'r'))

results = {}

query_list = []
query_id_list = []
doc_list = []
doc_id_list = []
for query_id, doc_ids in data.items():
    for doc_id in doc_ids:
        query = queries[query_id]
        doc = corpus[str(doc_id)]
        
        query_list.append(query) 
        doc_list.append(doc)

        query_id_list.append(query_id)
        doc_id_list.append(doc_id)

query_vectors = beir_splade.encode_queries(query_list, 32)
doc_vectors = beir_splade.encode_corpus(doc_list, 32)

assert len(query_vectors) == len(doc_vectors) == len(query_list) == len(doc_list)

with open('in-doc-negatives/outputs.tsv', 'w') as fo:
    fo.write("qid\tpid\tscore\n")
    for qid, pid, qv, dv in zip(query_id_list, doc_id_list, query_vectors, doc_vectors):
        score = qv.dot(dv).item()
        fo.write("\t".join(list(map(str, [qid, pid, score]))) + "\n")