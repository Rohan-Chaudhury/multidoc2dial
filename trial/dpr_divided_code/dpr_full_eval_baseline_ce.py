import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import concurrent.futures

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DPRQuestionEncoder, DPRContextEncoder
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import rankdata
import numpy as np
import torch.nn as nn


import json
import torch
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast,
    DPRReader, DPRReaderTokenizerFast, TrainingArguments, Trainer
)
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import transformers
from torch.optim import SGD
print(transformers.__version__)
from transformers import T5Tokenizer
import json
from torch.optim.lr_scheduler import CyclicLR
from transformers import DPRConfig, DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRReader, DPRReaderTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from tqdm.auto import tqdm
from transformers import T5EncoderModel
from transformers import AutoModel
from transformers import T5Config
from torch.nn.functional import cosine_similarity




import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"  # Set this to the index of the GPU you want to use
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import random

import json
import re
from datasets import Dataset

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_question(question):
    return remove_extra_spaces(question)


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.train.json", "r") as f:
    training_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.psg.multidoc2dial_all.structure.json", "r") as f:
    corpus_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.validation.json", "r") as f:
    validation_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.test.json", "r") as f:
    test_data = json.load(f)


import torch
import torch.nn.functional as F
from math import ceil
from tqdm import tqdm
import copy
import random
import datetime
import os


class T5CrossEncoder(nn.Module):
    def __init__(self, pretrained_model_name, model_max_length, dropout_rate=0.1):
        super().__init__()
        config = T5Config.from_pretrained(pretrained_model_name)
        config.model_max_length = model_max_length
        self.t5 = T5EncoderModel.from_pretrained(pretrained_model_name, config=config)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.classifier = self._init_classifier(config.d_model)

    def _init_classifier(self, d_model):
        classifier = nn.Linear(d_model, 1)
        return classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.layer_norm(pooled_output)
        logits = self.classifier(pooled_output)
        return logits






checkpoint_path_ce =  "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/66_ce/ce_checkpoint.pth"

def load_saved_model(checkpoint_path_ce):
    model_max_length = 1024
    checkpoint_ce = torch.load(checkpoint_path_ce, map_location=device)
    t5_pretrained_model_name = "t5-large"
    t5_cross_encoder = T5CrossEncoder(t5_pretrained_model_name, model_max_length)
    t5_cross_encoder= nn.DataParallel(t5_cross_encoder)
    t5_cross_encoder.to(device)
    t5_cross_encoder.load_state_dict(checkpoint_ce['cross_encoder_state_dict'])
    t5_cross_encoder.eval()
    return t5_cross_encoder

question_encoder = DPRQuestionEncoder.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
question_encoder = nn.DataParallel(question_encoder)
question_encoder = question_encoder.to(device)

question_encoder.eval()

context_encoder = DPRContextEncoder.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
context_encoder = nn.DataParallel(context_encoder)
context_encoder = context_encoder.to(device)

context_encoder.eval()



t5_cross_encoder = load_saved_model(checkpoint_path_ce)
# combined_model = load_saved_model(checkpoint_path_dpr)
print ("Model Loaded")

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_question(question):
    return remove_extra_spaces(question)


def preprocess_data(training_data):
    train_data = {
        "question": []
    }

    for item in training_data:
        question = preprocess_question(item["question"])


        train_data["question"].append(question)

    return train_data
# corpus_data_dict = preprocess_corpus_data(corpus_data)

preprocessed_data = preprocess_data(test_data)
train_dataset = Dataset.from_dict(preprocessed_data)


question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")


context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")


from tqdm import tqdm

def encode_questions(train_dataset, batch_size=1024):
    encoded_questions = []
    max_length = 512
    dataset_length = len(train_dataset)
    
    for start_idx in tqdm(range(0, dataset_length, batch_size), desc="Encoding questions"):
        end_idx = min(start_idx + batch_size, dataset_length)
        questions = [train_dataset[idx]["question"] for idx in range(start_idx, end_idx)]
        
        encodings = question_tokenizer(questions, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        input_ids = encodings["input_ids"].to(device)
        attention_masks = encodings["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = question_encoder(input_ids, attention_mask=attention_masks)
            embeddings = outputs.pooler_output.detach().cpu().numpy()
        
        encoded_questions.extend(embeddings)

    return np.vstack(encoded_questions)

# Assuming corpus_data_dict is your preprocessed data
print("Length of question_dataset:", len(train_dataset))

# Encode the preprocessed corpus data
question_embeddings = encode_questions(train_dataset)

print ("No.of questions: ",len(question_embeddings))



def preprocess_corpus_data(corpus_data):
    corpus_data_preprocessed = {
        "text": []
    }

    for item in corpus_data:
        text = remove_extra_spaces(item["text"])
        corpus_data_preprocessed["text"].append(text)
    
    return corpus_data_preprocessed

# corpus_data_dict = preprocess_corpus_data(corpus_data)

corpus_data_dict = preprocess_corpus_data(corpus_data)


def preprocess_corpus_list(corpus_data):
    corpus_data_list = []

    for item in corpus_data:
        text = remove_extra_spaces(item["text"])
        corpus_data_list.append(text)
    
    return corpus_data_list 


corpus_data_list = preprocess_corpus_list(corpus_data)
corpus_data_list = np.array(corpus_data_list)
def encode_context(corpus_dataset, batch_size=1024):
    encoded_contexts = []
    max_length = 512
    dataset_length = len(corpus_dataset)
    
    for start_idx in tqdm(range(0, dataset_length, batch_size), desc="Encoding contexts"):
        end_idx = min(start_idx + batch_size, dataset_length)
        contexts = [corpus_dataset[idx]["text"] for idx in range(start_idx, end_idx)]
        
        encodings = context_tokenizer(contexts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        input_ids = encodings["input_ids"].to(device)
        attention_masks = encodings["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = context_encoder(input_ids, attention_mask=attention_masks)
            embeddings = outputs.pooler_output.detach().cpu().numpy()
        
        encoded_contexts.extend(embeddings)

    return np.vstack(encoded_contexts)

# Assuming corpus_data_dict is your preprocessed data
corpus_dataset = Dataset.from_dict(corpus_data_dict)
print("Length of corpus_dataset:", len(corpus_dataset))

# Encode the preprocessed corpus data
context_embeddings = encode_context(corpus_dataset)

print("Length of embeddings:", len(context_embeddings))


index = faiss.IndexFlatIP(context_embeddings.shape[1])

index.add(context_embeddings)

print(f"Number of context embeddings in the FAISS index: {index.ntotal}")


def search_faiss(encoded_questions, index, k=10):
    D, I = index.search(encoded_questions, k)
    return I


def recall_at_k(predicted, positive_indices, k):
    num_correct = 0
    for pred, pos_idx in zip(predicted, positive_indices):
        if pos_idx in pred[:k]:
            num_correct += 1
    return num_correct / len(predicted)


def extract_positive_psg_ids(test_data):
    positive_psg_ids = []

    for item in test_data:
        positive_context_id = item["positive_ctxs"][0]["psg_id"]
        positive_psg_ids.append(positive_context_id)

    return positive_psg_ids

positive_psg_ids = extract_positive_psg_ids(test_data)
print ("Positive Passage Ids Extracted with length: ", len(positive_psg_ids))


# Search the FAISS index to find the top 10 most similar context embeddings
top_10_indices = search_faiss(question_embeddings, index, k=50)
t5_pretrained_model_name = "t5-large"
model_max_length = 1024
t5_tokenizer = T5Tokenizer.from_pretrained(t5_pretrained_model_name, model_max_length=model_max_length)



import torch



import math
from tqdm import tqdm

def rerank_with_cross_encoder(t5_questions, top_10_indices, context_passages, cross_encoder, tokenizer, k=10, batch_size=2):
    max_length = 1024
    num_batches = math.ceil(len(t5_questions) / batch_size)
    reranked_indices = []

    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(t5_questions))

        batch_questions = t5_questions[batch_start:batch_end]
        batch_top_indices = top_10_indices[batch_start:batch_end]

        t5_input = [f"{question} </s> {context_passages[top_idx]}" for question, top_indices in zip(batch_questions, batch_top_indices) for top_idx in top_indices]
        
        t5_encodings = tokenizer(t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
        t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)

        with torch.no_grad():
            logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze().detach().cpu().numpy()
            logits = logits.reshape(len(batch_questions), -1)

        batch_reranked_indices = [top_indices[np.argsort(logits[i])[::-1][:k]] for i, top_indices in enumerate(batch_top_indices)]
        reranked_indices.extend(batch_reranked_indices)

    return np.array(reranked_indices)


# Calculate recall@1, recall@5, recall@10 scores
recall_1 = recall_at_k(top_10_indices, positive_psg_ids, 1)
recall_5 = recall_at_k(top_10_indices, positive_psg_ids, 5)
recall_10 = recall_at_k(top_10_indices, positive_psg_ids, 10)

print(f"Recall@1: {recall_1:.4f}")
print(f"Recall@5: {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")



# Search using cosine similarity to find the top 10 most similar context embeddings

import numpy as np
import pickle
def load_list_from_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
t5_questions = load_list_from_file('t5_questions_500.pkl')


reranked_indices_cosine = rerank_with_cross_encoder(t5_questions, top_10_indices, corpus_data_list , t5_cross_encoder, t5_tokenizer, k=10)
# Calculate recall@1, recall@5, recall@10 scores
recall_1_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 1)
recall_5_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 5)
recall_10_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 10)

print(f"Recall@1 (Cosine): {recall_1_cosine:.4f}")
print(f"Recall@5 (Cosine): {recall_5_cosine:.4f}")
print(f"Recall@10 (Cosine): {recall_10_cosine:.4f}")
