#FROM Sentence-BERT(https://github.com/UKPLab/sentence-transformers/blob/afee883a17ab039120783fd0cffe09ea979233cf/examples/training/ms_marco/train_bi-encoder_margin-mse.py) with minimal changes.
#Original License Apache2, NOTE: Trained MSMARCO models are NonCommercial (from dataset License)
import logging
from multiprocessing.sharedctypes import Value

from transformers import AdamW
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, InputExample
import models
from datetime import datetime
import torch.nn as nn
import gzip
import os
import tarfile
import tqdm
from torch.utils.data import Dataset
import random
from shutil import copyfile
import pickle
import argparse
import losses
import torch
from beir.retrieval.train import TrainRetriever
from beir.datasets.data_loader import GenericDataLoader


#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=32, type=int)
parser.add_argument("--max_seq_length", default=256, type=int)
parser.add_argument("--model_name", default="distilbert-base-uncased", type=str)
parser.add_argument("--data_path", type=str)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used ? Multiple systems seperated by comma. None = all")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lambda_d", default=0.08, type=float)
parser.add_argument("--lambda_q", default=0.1, type=float)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=20, type=int)
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--pair_lambda", type=float, default=0.0)
args = parser.parse_args()
device = torch.device('cuda:0')
#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logging.info(str(args))

train_batch_size = args.train_batch_size  # Increasing the train batch size generally improves the model performance, but requires more GPU memory
model_name = args.model_name
max_passages = args.max_passages
max_seq_length = args.max_seq_length  # Max length for passages. Increasing it implies more GPU memory needed
num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
num_epochs = args.epochs  # Number of epochs we want to train

# Load our embedding model
logging.info("Create new SBERT model")
word_embedding_model = models.MLMTransformer(model_name, max_seq_length=max_seq_length)
model = SentenceTransformer(modules=[word_embedding_model])
# if torch.cuda.device_count() > 1:
#     model = nn.DataParallel(model)  # Change device_ids as per your available GPUs
model.to(device)
# Move the model to the specified device

model_save_path = f'output/distilsplade_max_{args.lambda_q}_{args.lambda_d}_{model_name.replace("/", "-")}-batch_size_{train_batch_size}-lr_{args.lr}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

# Write self to path
os.makedirs(model_save_path, exist_ok=True)

train_script_path = os.path.join(model_save_path, 'train_script.py')
copyfile(__file__, train_script_path)
with open(train_script_path, 'a') as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

### Now we read the MS MARCO dataset
corpus, queries, qrels = GenericDataLoader(args.data_path).load(split='train')


train_queries = {}

for query_id in qrels.keys():
    positives = []
    negatives = []
    for doc_id, label in qrels[query_id].items():
        if label == 0:
            negatives.append(doc_id)
        elif label == 1:
            positives.append(doc_id)
        else:
            raise ValueError(f'unknown label: {label}')
    assert len(positives) > 0 and len(negatives) > 0, f"query {query_id}, text: {queries[query_id]}, pos_len: {len(positives)}, neg_len: {len(negatives)}"
    train_queries[query_id] = {"qid": query_id, 'query': queries[query_id], 'pos': positives, 'neg': negatives}

logging.info("Train queries: {}".format(len(train_queries)))



# We create a custom MS MARCO dataset that returns triplets (query, positive, negative)
# on-the-fly based on the information from the mined-hard-negatives jsonl file.
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=model.smart_batching_collate)

# for i, batch in enumerate(train_dataloader):
#     batch = [t.to(device) for t in batch]
if args.pair_lambda > 0:
    train_loss = losses.MultipleNegativesRankingLossSpladePair(model=model, lambda_q=args.lambda_q, lambda_d=args.lambda_d, pair_lambda=args.pair_lambda)
else:
    train_loss = losses.MultipleNegativesRankingLossSplade(model=model, lambda_q=args.lambda_q, lambda_d=args.lambda_d)


# if torch.cuda.device_count() > 1:
#     train_objectives = [(nn.DataParallel(model), train_loss)]
# else:
train_objectives = [(train_dataloader, train_loss)]

# train_dataloader.to(device)
# Train the model
model.fit(train_objectives=train_objectives,
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=len(train_dataloader),
          optimizer_params = {'lr': args.lr},
          save_best_model=True)

# optimizer = AdamW(model.parameters(), lr=args.lr)
# # Set up the training loop
# num_accumulation_steps = 64
# num_batches = len(train_dataloader)

# for epoch in range(num_epochs):
#     total_loss = 0

#     for i, batch in enumerate(train_dataloader):
#         # Get the input data and labels from the batch
#         # print(batch[1])
#         input_ids = batch[0][0]['input_ids']
#         attention_mask = batch[0][0]['attention_mask']
#         labels = batch[1]

#         # Compute the loss for this batch
#         outputs = model()
#         loss = outputs.loss

#         # Accumulate the loss over multiple batches
#         loss = loss / num_accumulation_steps
#         total_loss += loss.item()

#         # Backpropagate the loss and update the model weights
#         loss.backward()

#         if (i+1) % num_accumulation_steps == 0:
#             optimizer.step()
#             optimizer.zero_grad()

#         # Log the loss every 1000 batches
#         if (i+1) % 1000 == 0:
#             print(f"Epoch {epoch + 1}, batch {i+1}/{num_batches}, loss: {total_loss / 1000}")
#             total_loss = 0

#     # Save the model after each epoch
#     if model_save_path is not None:
#         model.save_pretrained(model_save_path)

# Save model
model.save(model_save_path)