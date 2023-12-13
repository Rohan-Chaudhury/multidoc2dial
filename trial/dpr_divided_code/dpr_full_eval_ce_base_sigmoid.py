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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set this to the index of the GPU you want to use
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

class CustomDPRContextEncoder(nn.Module):
    def __init__(self, model_name, dropout_rate):
        super(CustomDPRContextEncoder, self).__init__()
        self.model = DPRContextEncoder.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.linear = self._init_linear(self.model.config.hidden_size)

    def _init_linear(self, hidden_size):
        linear = nn.Linear(hidden_size, hidden_size)
        return linear

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # print (linear_output.shape)
        # print (pooled_output.shape)
        return pooled_output




class CustomDPRQuestionEncoderWithDropout(nn.Module):
    def __init__(self, model_name, dropout_rate):
        super(CustomDPRQuestionEncoderWithDropout, self).__init__()
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.linear = self._init_linear(self.model.config.hidden_size)

    def _init_linear(self, hidden_size):
        linear = nn.Linear(hidden_size, hidden_size)
        return linear

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output




class DPRCombinedModel(nn.Module):
    def __init__(self, question_encoder: CustomDPRQuestionEncoderWithDropout, context_encoder: CustomDPRContextEncoder):
        super(DPRCombinedModel, self).__init__()
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder

    def forward(self, question_input_ids, question_attention_mask, context_input_ids, context_attention_mask):
        question_outputs = self.question_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask)
        context_outputs = self.context_encoder(input_ids=context_input_ids, attention_mask=context_attention_mask)
        return question_outputs, context_outputs

import torch
import torch.nn.functional as F
from math import ceil
from tqdm import tqdm
import copy
import random
import datetime
import os


# class T5CrossEncoder(nn.Module):
#     def __init__(self, pretrained_model_name, model_max_length, dropout_rate=0.0, fine_tune=False, trainable_layers=1):
#         super().__init__()
#         config = T5Config.from_pretrained(pretrained_model_name)
#         config.model_max_length = model_max_length
#         self.t5 = T5EncoderModel.from_pretrained(pretrained_model_name, config=config)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.attention = nn.MultiheadAttention(config.d_model, num_heads=16)
#         self.residual = nn.Sequential(
#             nn.Linear(config.d_model, config.d_model),
#             nn.LayerNorm(config.d_model),
#         )
#         self.classifier = self._init_classifier(config.d_model)

#         # Freeze layers except the last 'trainable_layers'
#         if not fine_tune:
#             if trainable_layers < len(self.t5.encoder.block):
#                 for layer in self.t5.encoder.block[:-trainable_layers]:
#                     for param in layer.parameters():
#                         param.requires_grad = False

#     def _init_classifier(self, d_model):
#         classifier = nn.Sequential(
#             nn.Linear(d_model, d_model),
#             nn.GELU(),
#             nn.LayerNorm(d_model),
#             nn.Dropout(0.0),
#             nn.Linear(d_model, d_model // 2),
#             nn.GELU(),
#             nn.LayerNorm(d_model // 2),
#             nn.Dropout(0.0),
#             nn.Linear(d_model // 2, 1),
#             nn.Sigmoid(),
#         )
#         for module in classifier:
#             if isinstance(module, nn.Linear):
#                 nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
#         return classifier

#     def forward(self, input_ids, attention_mask):
#         outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         last_hidden_state = outputs.last_hidden_state

#         # Apply multi-head self-attention
#         attention_output, _ = self.attention(last_hidden_state, last_hidden_state, last_hidden_state, need_weights=False)
#         attention_output = self.residual(attention_output) + last_hidden_state
        
#         pooled_output = torch.mean(attention_output, dim=1)
        
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         return logits


# class T5CrossEncoder(nn.Module):
#     def __init__(self, pretrained_model_name, model_max_length, dropout_rate=0.0, fine_tune=True):
#         super().__init__()
#         config = T5Config.from_pretrained(pretrained_model_name)
#         config.model_max_length = model_max_length
#         self.t5 = T5EncoderModel.from_pretrained(pretrained_model_name, config=config)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.layer_norm = nn.LayerNorm(config.d_model)
#         self.attention = nn.Linear(config.d_model, 1)
#         self.classifier = self._init_classifier(config.d_model)

#         # if not fine_tune:
#         #     for param in self.t5.parameters():
#         #         param.requires_grad = False

#     def _init_classifier(self, d_model):
#         classifier = nn.Linear(d_model, 1)
#         nn.init.kaiming_normal_(classifier.weight, nonlinearity='relu')
#         return classifier
    
#     def forward(self, input_ids, attention_mask):
#         outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         last_hidden_state = outputs.last_hidden_state

#         # Assume attention_mask is 1 for real tokens and 0 for padding
#         attention_scores = self.attention(last_hidden_state)
#         attention_scores = attention_scores.masked_fill_(attention_mask.unsqueeze(-1) == 0, float('-inf'))
#         weights = F.softmax(attention_scores, dim=1)

#         pooled_output = torch.sum(weights * last_hidden_state, dim=1)
        
#         pooled_output = self.dropout(pooled_output)
#         pooled_output = self.layer_norm(pooled_output)
#         logits = self.classifier(pooled_output)
#         return logits


class T5CrossEncoder(nn.Module):
    def __init__(self, pretrained_model_name, model_max_length, dropout_rate=0.0):
        super().__init__()
        config = T5Config.from_pretrained(pretrained_model_name)
        config.model_max_length = model_max_length
        self.t5 = T5EncoderModel.from_pretrained(pretrained_model_name, config=config)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.MultiheadAttention(config.d_model, num_heads=32)
        self.classifier = self._init_classifier(config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)



    def _init_classifier(self, d_model):
        classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model // 2),
            nn.Linear(d_model // 2, 1),
        )
        for module in classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        return classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state

        # Create a mask for the multihead attention
        # We want to ignore padding tokens, so we set these positions to `-inf`
        # Non-padding positions are set to `0`
        attention_mask_for_multihead = (1.0 - attention_mask) * -1e9  # size: [batch_size, seq_len]
        attention_mask_for_multihead = attention_mask_for_multihead.to(last_hidden_state.dtype)

        # Expand the attention_mask_for_multihead to (batch_size, seq_len, seq_len)
        seq_len = last_hidden_state.shape[1]
        attention_mask_for_multihead = attention_mask_for_multihead.unsqueeze(1)  # size: [batch_size, 1, seq_len]
        attention_mask_for_multihead = attention_mask_for_multihead.expand(-1, seq_len, -1)  # size: [batch_size, seq_len, seq_len]
        # Compute num_heads * batch_size
        num_heads_times_batch_size = 32  # replace '16' with the actual number of heads
        # Repeat the attention_mask_for_multihead along a new dimension
        attention_mask_for_multihead = attention_mask_for_multihead.repeat(num_heads_times_batch_size, 1, 1)

        # Transpose last_hidden_state for nn.MultiheadAttention
        last_hidden_state = last_hidden_state.transpose(0, 1)

        # Apply multi-head self-attention
        attention_output, _ = self.attention(last_hidden_state, last_hidden_state, last_hidden_state,
                                              attn_mask=attention_mask_for_multihead, need_weights=False)
        attention_output = self.layer_norm(attention_output + last_hidden_state)  # residual connection and LayerNorm
        attention_output = attention_output.transpose(0, 1)
        pooled_output = torch.mean(attention_output, dim=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


checkpoint_path_dpr = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/best_model_state.bin"
checkpoint_path_ce = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/guardian/ce_checkpoint.pth"

print ("Checkpoint path: ", checkpoint_path_dpr)
print ("Checkpoint path: ", checkpoint_path_ce)

def load_saved_model(checkpoint_path_dpr, checkpoint_path_ce):
    question_encoder = CustomDPRQuestionEncoderWithDropout("sivasankalpp/dpr-multidoc2dial-structure-question-encoder", 0.0)
    context_encoder = CustomDPRContextEncoder(model_name="sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder", dropout_rate=0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_encoder = nn.DataParallel(question_encoder)
    context_encoder = nn.DataParallel(context_encoder)

    question_encoder.to(device)
    question_encoder.eval()
    context_encoder.to(device)
    context_encoder.eval()
    combined_model = DPRCombinedModel(question_encoder, context_encoder)


    combined_model.load_state_dict(torch.load(checkpoint_path_dpr , map_location=torch.device('cuda')), strict=False)
    combined_model.to(device)
    combined_model.eval()

    model_max_length = 1024
    checkpoint_ce = torch.load(checkpoint_path_ce, map_location=device)
    print ("CE Loss: ", checkpoint_ce["loss"])
    t5_pretrained_model_name = "t5-base"
    t5_cross_encoder = T5CrossEncoder(t5_pretrained_model_name, model_max_length)
    t5_cross_encoder= nn.DataParallel(t5_cross_encoder)
    t5_cross_encoder.to(device)
    t5_cross_encoder.load_state_dict(checkpoint_ce['cross_encoder_state_dict'])
    # print ("Loss: ", checkpoint_ce["loss"])
    t5_cross_encoder.eval()
    return combined_model, t5_cross_encoder

combined_model, t5_cross_encoder = load_saved_model(checkpoint_path_dpr, checkpoint_path_ce)
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

def encode_questions(train_dataset, batch_size=256*2):
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
            outputs = combined_model.question_encoder(input_ids, attention_mask=attention_masks)
            embeddings = outputs.detach().cpu().numpy()
        
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
def encode_context(corpus_dataset, batch_size=256*2):
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
            outputs = combined_model.context_encoder(input_ids, attention_mask=attention_masks)
            embeddings = outputs.detach().cpu().numpy()
        
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


# Train the quantizer
# nlist = 100  # Number of Voronoi cells; adjust this value based on your dataset
# quantizer = faiss.IndexFlatIP(context_embeddings.shape[1])
# index = faiss.IndexIVFFlat(quantizer, context_embeddings.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)

# # Train the index
# index.train(context_embeddings)

# # Add context embeddings to the index
# index.add(context_embeddings)

# print(f"Number of context embeddings in the FAISS index: {index.ntotal}")

# def search_faiss(encoded_questions, index, k=10):
#     D, I = index.search(encoded_questions, k)
#     return I


# 3. Calculate recall@1, recall@5, recall@10 scores
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
top_10_indices = search_faiss(question_embeddings, index, k=10)
t5_pretrained_model_name = "t5-base"
model_max_length = 1024
t5_tokenizer = T5Tokenizer.from_pretrained(t5_pretrained_model_name, model_max_length=model_max_length)


# def rerank_with_cross_encoder(t5_questions, top_10_indices, context_passages, cross_encoder, tokenizer, k=10):
#     reranked_indices = []
#     max_length = 1024
#     for question, top_indices in tqdm(zip(t5_questions, top_10_indices)):
#         top_contexts = context_passages[top_indices]
#         t5_input = [f"{question} </s> {ctx}" for ctx in top_contexts]
#         t5_encodings = tokenizer(t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
#         t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)

#         with torch.no_grad():
#             logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze().detach().cpu().numpy()
#             reranked_top_indices = top_indices[np.argsort(logits)[::-1][:k]]

        
#         reranked_indices.append(reranked_top_indices)
    
#     return np.array(reranked_indices)



import torch

# def rerank_with_cross_encoder(t5_questions, top_10_indices, context_passages, cross_encoder, tokenizer, k=10):
#     max_length = 1024
#     batch_size = len(t5_questions)
#     top_k = top_10_indices.shape[1]

#     # Expand and repeat the questions and top_indices tensors
#     repeated_questions = t5_questions.repeat_interleave(top_k)
#     repeated_top_indices = top_10_indices.repeat(1, top_k).view(-1)

#     # Get the top context passages
#     top_contexts = context_passages[repeated_top_indices]

#     # Create the T5 input pairs
#     t5_input = [f"{q} </s> {ctx}" for q, ctx in zip(repeated_questions, top_contexts)]

#     # Tokenize the input pairs
#     t5_encodings = tokenizer(t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
#     t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)

#     with torch.no_grad():
#         # Get logits from cross_encoder
#         logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()

#         # Reshape logits to (batch_size, top_k)
#         logits = logits.view(batch_size, top_k)

#         # Get the top k indices after reranking
#         reranked_top_indices = top_10_indices[torch.arange(batch_size).unsqueeze(-1), logits.argsort(descending=True)[:, :k]]

#     return reranked_top_indices.cpu().numpy()


import math
from tqdm import tqdm

def rerank_with_cross_encoder(t5_questions, top_10_indices, context_passages, cross_encoder, tokenizer, k=10, batch_size=1):
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




from sklearn.metrics.pairwise import cosine_similarity

# def search_cosine_similarity(encoded_questions, context_embeddings, k=10):
#     similarity_matrix = cosine_similarity(encoded_questions, context_embeddings)
#     I = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]
#     return I


import numpy as np

# def search_cosine_similarity(encoded_questions, context_embeddings, k=10, alpha=0.5):
#     # Normalize the embeddings
#     norm_encoded_questions = encoded_questions / np.linalg.norm(encoded_questions, axis=1, keepdims=True)
#     norm_context_embeddings = context_embeddings / np.linalg.norm(context_embeddings, axis=1, keepdims=True)

#     # Compute the cosine similarity (dot product)
#     cosine_similarity_matrix = np.dot(norm_encoded_questions, norm_context_embeddings.T)

#     # Compute the Euclidean distance
#     euclidean_distance_matrix = np.linalg.norm(encoded_questions[:, np.newaxis] - context_embeddings, axis=2)

#     # Normalize the Euclidean distance matrix to the range [0, 1]
#     normalized_euclidean_distance_matrix = euclidean_distance_matrix / np.max(euclidean_distance_matrix)

#     # Compute the combined similarity measure
#     combined_similarity_matrix = alpha * cosine_similarity_matrix + (1 - alpha) * (1 - normalized_euclidean_distance_matrix)

#     # Get the top k indices
#     I = np.argsort(combined_similarity_matrix, axis=1)[:, ::-1][:, :k]
#     return I


# import torch

# def search_cosine_similarity(encoded_questions, context_embeddings, k=10):
#     # Convert numpy arrays to PyTorch tensors
#     encoded_questions_tensor = torch.tensor(encoded_questions, dtype=torch.float)
#     context_embeddings_tensor = torch.tensor(context_embeddings, dtype=torch.float)

#     # Compute the cosine similarity
#     similarity_matrix = torch.nn.functional.cosine_similarity(
#         encoded_questions_tensor.unsqueeze(1), context_embeddings_tensor.unsqueeze(0), dim=-1
#     )

#     # Convert the similarity matrix back to a numpy array
#     similarity_matrix = similarity_matrix.numpy()

#     # Get the top k indices
#     I = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]
#     return I


import numpy as np

# def search_cosine_similarity(encoded_questions, context_embeddings, k=30):
#     # Compute the dot product (matrix multiplication) between the two sets of embeddings
#     similarity_matrix = np.dot(encoded_questions, context_embeddings.T)

#     # Get the top k indices
#     I = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]
#     return I


def search_cosine_similarity(encoded_questions, context_embeddings, k=50):
    similarity_matrix = cosine_similarity(encoded_questions, context_embeddings)
    I = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]
    return I


# Search using cosine similarity to find the top 10 most similar contextd embeddings
top_10_indices_cosine = search_cosine_similarity(question_embeddings, context_embeddings, k=50)
import pickle

def load_list_from_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Calculate recall@1, recall@5, recall@10 scores
recall_1 = recall_at_k(top_10_indices_cosine, positive_psg_ids, 1)
recall_5 = recall_at_k(top_10_indices_cosine, positive_psg_ids, 5)
recall_10 = recall_at_k(top_10_indices_cosine, positive_psg_ids, 10)

print(f"Recall@1: {recall_1:.4f}")
print(f"Recall@5: {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")


# t5_questions = load_list_from_file('t5_questions_500.pkl')
t5_questions = train_dataset['question']



reranked_indices_cosine = rerank_with_cross_encoder(t5_questions, top_10_indices_cosine, corpus_data_list , t5_cross_encoder, t5_tokenizer, k=10)
# Calculate recall@1, recall@5, recall@10 scores
recall_1_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 1)
recall_5_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 5)
recall_10_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 10)

print(f"Recall@1 (Cosine): {recall_1_cosine:.4f}")
print(f"Recall@5 (Cosine): {recall_5_cosine:.4f}")
print(f"Recall@10 (Cosine): {recall_10_cosine:.4f}")


reranked_indices_cosine = rerank_with_cross_encoder(t5_questions, reranked_indices_cosine, corpus_data_list , t5_cross_encoder, t5_tokenizer, k=10)
# Calculate recall@1, recall@5, recall@10 scores
recall_1_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 1)
recall_5_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 5)
recall_10_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 10)

print(f"Recall@1 (Cosine): {recall_1_cosine:.4f}")
print(f"Recall@5 (Cosine): {recall_5_cosine:.4f}")
print(f"Recall@10 (Cosine): {recall_10_cosine:.4f}")
