import sys
import pytrec_eval
from beir import util
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from typing import Type, List, Dict, Union, Tuple
from models import Splade, BEIRSpladeModel, BEIRDPR, t5Embedder
from beir.retrieval.custom_metrics import mrr
from transformers import AutoTokenizer, DPRQuestionEncoder, DPRContextEncoder, T5Tokenizer, BertModel, AutoModel


import argparse

import pdb

BERT_MODEL = "mrm8488/t5-base-finetuned-break_data-question-retrieval"
# BERT_VOCAB = "bert_model/scibert_scivocab_uncased/vocab.txt"
#BERT_MODEL = "allenai/scibert_scivocab_uncased"


config = {
    "bert_model": BERT_MODEL,
    "bert_trainable": False,
    "cacheable_tasks": [],

    "dropout": 0.5,
    "word_lstm_hs": 758,
    "att_pooling_dim_ctx": 200,
    "att_pooling_num_ctx": 15,

    "lr": 3e-05,
    "lr_epoch_decay": 0.9,
    "batch_size":  1,
    "max_seq_length": 128,
    "max_epochs": 20,
    "early_stopping": 5,

}

def evaluate(qrels: Dict[str, Dict[str, int]],
             results: Dict[str, Dict[str, float]],
             k_values: List[int]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    _mrr = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)

    _mrr = mrr(qrels, results, k_values)

    for eval in [ndcg, _map, recall, precision, _mrr]:
        for k in eval.keys():
            print("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision, _mrr

def recall_at_k(qrels, results, k=10):
    not_retreived = []
    recall = 0
    counts = 0
    for query_id in qrels.keys():
        results_at_k = [doc_id for (doc_id, score) in sorted(results[query_id].items(), key=lambda x: x[1], reverse=True)][:k]
        results_at_k = set(results_at_k)
        for doc_id in qrels[query_id].keys():
            if doc_id in results_at_k:
                recall += 1
            else:
                not_retreived.append((query_id, doc_id))
            counts += 1

    return recall/counts, not_retreived

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", default=None)

    args = parser.parse_args()
    return args

corpus, queries, qrels = GenericDataLoader(
    "../retrieval_data/").load(split="dev")

print("lengths of queries, corpus, qrels:", len(queries), len(corpus), len(qrels))

args = get_args()
model_name = args.model_name
model_path = args.model_path

if "dpr-ft" in model_name:
    if model_path is None:
        query_encoder = DPRQuestionEncoder.from_pretrained(
            "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
        query_tokenizer = AutoTokenizer.from_pretrained(
            "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")

        doc_encoder = DPRContextEncoder.from_pretrained(
            "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
        doc_tokenizer = AutoTokenizer.from_pretrained(
            "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
    else:
        query_encoder = DPRQuestionEncoder.from_pretrained(
            model_path)
        query_tokenizer = AutoTokenizer.from_pretrained(
            model_path)

        doc_encoder = DPRContextEncoder.from_pretrained(
            model_path)
        doc_tokenizer = AutoTokenizer.from_pretrained(
            model_path)

    beir_model = BEIRDPR(query_encoder, doc_encoder,
                         query_tokenizer, doc_tokenizer)
    model = DRES(beir_model, batch_size=128)

if "deepset" in model_name:
    if model_path is None:
        query_encoder = AutoModel.from_pretrained(
            "facebook/spar-wiki-bm25-lexmodel-context-encoder")
        query_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/spar-wiki-bm25-lexmodel-context-encoder")

        doc_encoder = AutoModel.from_pretrained(
            "facebook/spar-wiki-bm25-lexmodel-context-encoder")
        doc_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/spar-wiki-bm25-lexmodel-context-encoder")
        # query_encoder = DPRQuestionEncoder.from_pretrained(
        #     "vblagoje/dpr-ctx_encoder-single-lfqa-wiki")
        # query_tokenizer = AutoTokenizer.from_pretrained(
        #     "vblagoje/dpr-ctx_encoder-single-lfqa-wiki")

        # doc_encoder = DPRContextEncoder.from_pretrained(
        #     "vblagoje/dpr-ctx_encoder-single-lfqa-wiki")
        # doc_tokenizer = AutoTokenizer.from_pretrained(
        #     "vblagoje/dpr-ctx_encoder-single-lfqa-wiki")
    else:
        query_encoder = AutoModel.from_pretrained(
            model_path)
        query_tokenizer = AutoTokenizer.from_pretrained(
            model_path)

        doc_encoder = AutoModel.from_pretrained(
            model_path)
        doc_tokenizer = AutoTokenizer.from_pretrained(
            model_path)

    beir_model = BEIRDPR(query_encoder, doc_encoder,
                         query_tokenizer, doc_tokenizer)
    model = DRES(beir_model, batch_size=128)

elif "dpr-nq" in model_name:
    if model_path is None:
        model = DRES(models.SentenceBERT(
            "sentence-transformers/facebook-dpr-question_encoder-multiset-base"), batch_size=128)
    else:
        model = DRES(models.SentenceBERT(
            model_path), batch_size=128)

elif "tas-b" in model_name:
    if model_path is None:
        model = DRES(models.SentenceBERT(
            "sentence-transformers/msmarco-distilbert-base-tas-b"), batch_size=128)
    else:
        model = DRES(models.SentenceBERT(
            model_path), batch_size=128)

elif "distilsplade" in model_name:
    if model_path is None:
        model_type_or_dir = "../splade/distilsplade_max"
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)
    else:
        model_type_or_dir = model_path
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)

elif "d1" in model_name:
    if model_path is None:
        model_type_or_dir = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/md2d/splade/training_with_sentence_transformers/output/distilsplade_max_0.1_0.08_-home-grads-r-rohan.chaudhury-multidoc2dial-multidoc2dial-splade-distilsplade_max-batch_size_16-lr_1e-06-2023-02-20_21-13-37/0_MLMTransformer"
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)
    else:
        model_type_or_dir = model_path
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)

elif "d2" in model_name:
    if model_path is None:
        model_type_or_dir = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/md2d/splade/training_with_sentence_transformers/output/distilsplade_max_0.1_0.08_-home-grads-r-rohan.chaudhury-multidoc2dial-multidoc2dial-splade-splade_distil_CoCodenser_large-batch_size_16-lr_1e-06-2023-02-21_05-49-36/0_MLMTransformer"
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)
    else:
        model_type_or_dir = model_path
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)


elif "d3" in model_name:
    if model_path is None:
        model_type_or_dir = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/md2d/splade/training_with_sentence_transformers/output/distilsplade_max_0.1_0.08_-home-grads-r-rohan.chaudhury-multidoc2dial-multidoc2dial-splade-splade_max_CoCodenser-batch_size_64-lr_1e-06-2023-02-22_04-00-55/0_MLMTransformer"
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)
    else:
        model_type_or_dir = model_path
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)

elif "d4" in model_name:
    if model_path is None:
        model_type_or_dir = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/md2d/splade/training_with_sentence_transformers/output/distilsplade_max_0.1_0.08_-home-grads-r-rohan.chaudhury-multidoc2dial-multidoc2dial-splade-splade_max_CoCodenser-batch_size_32-lr_2e-05-2023-02-23_00-10-34/20130/0_MLMTransformer"
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)
    else:
        model_type_or_dir = model_path
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)


elif "splade" in model_name:
    if model_path is None:
        model_type_or_dir = "../splade/splade_max_distilbert"
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)
    else:
        model_type_or_dir = model_path
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)

elif "spladeCo" in model_name:
    if model_path is None:
        model_type_or_dir = "../splade/splade_max_CoCodenser"
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)
    else:
        model_type_or_dir = model_path
        model = Splade(model_type_or_dir)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        beir_splade = BEIRSpladeModel(model, tokenizer)
        model = DRES(beir_splade)

elif "t5" in model_name:
    model= t5Embedder(BERT_MODEL)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(BERT_MODEL)
    beir_t5 = BEIRSpladeModel(model, tokenizer)
    model = DRES(beir_t5)


retriever = EvaluateRetrieval(model, score_function="dot", k_values=[9]) # retriever retrieves topk +1 for some reason
results = retriever.retrieve(corpus, queries)

# assert len(results) == len(queries)
# for query_id in results.keys():
    # assert query_id in qrels
    # assert len(results[query_id]) == 10, f"{len(results[query_id])}"
    # for doc_id,score in results[query_id].items():
    #     assert doc_id in corpus

_mrr = mrr(qrels, results, [10])
recall, not_retrieved = recall_at_k(qrels, results)

ndcg, _map, recall, precision, _mrr = evaluate(qrels, results, [1,5,10])
print(ndcg, _map, recall, precision, _mrr)

with open(f'../retrieval_data/{model_name}-results.tsv', 'w') as fo:
    for query_id in results.keys():
        for doc_id, score in sorted(results[query_id].items(), key=lambda x: x[1], reverse=True):
            fo.write(
                '\t'.join(list(map(str, [query_id, doc_id, score]))) + '\n')

with open(f'../retrieval_data/{model_name}-not_retrieved.tsv', 'w') as fo:
    for query_id, doc_id in not_retrieved:
        fo.write('\t'.join([str(query_id), str(doc_id)]) + '\n')

print(len(results))

# python run_beir.py splade
# python run_beir.py dpr
# python run_beir.py tas-b
# python run_beir.py distilsplade