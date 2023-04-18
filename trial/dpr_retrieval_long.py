
import torch
from torch.utils.data import DataLoader
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from typing import List, Dict
import numpy as np
import json
import elasticsearch
from elasticsearch import Elasticsearch
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from evaluation_metrics import f1_score, mrr_score, recall_at_k, exact_match, bleu_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
checkpoint_path = "path/to/checkpoints"
question_encoder = DPRQuestionEncoder.from_pretrained(checkpoint_path).to(device)
context_encoder = DPRContextEncoder.from_pretrained(checkpoint_path).to(device)

# Initialize tokenizers
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

model = DPRCombinedModel(question_encoder, context_encoder)

# Load test dataset and corpus
with open("path/to/test_dataset.json", "r") as file:
    test_data = json.load(file)

with open("path/to/corpus.json", "r") as file:
    corpus = json.load(file)

# Elasticsearch setup
es = Elasticsearch()
INDEX_NAME = "dpr_embeddings"
if not es.indices.exists(INDEX_NAME):
    es.indices.create(index=INDEX_NAME)

# Function to index passages in Elasticsearch
def index_passages(passages: List[Dict], model, tokenizer, es, index_name):
    for passage in passages:
        passage_input = tokenizer(passage["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        passage_embedding = model.context_encoder(passage_input.to(device)).cpu().detach().numpy()
        es.index(index=index_name, id=passage["id"], body={"text": passage["text"], "embedding": passage_embedding.tolist()})

index_passages(corpus, model, context_tokenizer, es, INDEX_NAME)

# Function to search passages in Elasticsearch
def search_passages(query_embedding, es, index_name, top_k=10):
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_embedding, 'embedding') + 1.0",
                "params": {"query_embedding": query_embedding},
            },
        }
    }
    res = es.search(index=index_name, body={"query": script_query, "size": top_k})
    return [(hit["_id"], hit["_score"]) for hit in res["hits"]["hits"]]

# Evaluation
num_top_k = [1, 5, 10]
metrics = defaultdict(list)

for sample in test_dataset:
    question = sample["question"]
    ground_truth_ids = [ctx["psg_id"] for ctx in sample["positive_ctxs"]]

# Encode question
tokenized_question = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
question_input_ids, question_attention_mask = tokenized_question["input_ids"], tokenized_question["attention_mask"]

# Score passages in corpus
scores = []
for passage in corpus:
    tokenized_passage = tokenizer(passage["text"], return_tensors="pt", padding=True, truncation=True, max_length=128)
    context_input_ids, context_attention_mask = tokenized_passage["input_ids"], tokenized_passage["attention_mask"]

    question_outputs, context_outputs = model(
        question_input_ids=question_input_ids,
        question_attention_mask=question_attention_mask,
        context_input_ids=context_input_ids,
        context_attention_mask=context_attention_mask
    )

    score = (question_outputs @ context_outputs.T).item()
    scores.append((passage["id"], score))

# Sort and retrieve top k passages
scores = sorted(scores, key=lambda x: x[1], reverse=True)

for k in num_top_k:
    top_k_ids = [x[0] for x in scores[:k]]
    metrics[f"r@{k}"].append(any(gt_id in top_k_ids for gt_id in ground_truth_ids))

# Calculate MRR
rank = 0
for idx, (psg_id, _) in enumerate(scores):
    if psg_id in ground_truth_ids:
        rank = idx + 1
        break
metrics["mrr"].append(1 / rank)

# Calculate EM, F1, and BLEU
top_answer = corpus[scores[0][0]]["text"]
reference_answers = sample["answers"]
prediction = top_answer

em_score = max([exact_match_score(prediction, ans) for ans in reference_answers])
f1_score = max([f1_score(prediction, ans) for ans in reference_answers])
bleu_score = corpus_bleu([[ans] for ans in reference_answers], [prediction]).score

metrics["em"].append(em_score)
metrics["f1"].append(f1_score)
metrics["bleu"].append(bleu_score)


# Calculate average scores
average_metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}
print(average_metrics)

# Save results to a JSON file
import json

with open("evaluation_results.json", "w") as f:
    json.dump(average_metrics, f, indent=4)

# Print results in a readable format
print("Evaluation Results:")
for metric, score in average_metrics.items():
    print(f"{metric}: {score:.4f}")
