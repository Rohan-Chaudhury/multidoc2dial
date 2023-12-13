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
import torch
import torch.nn as nn
from transformers import DPRQuestionEncoder
from torch.nn import MultiheadAttention

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"  # Set this to the index of the GPU you want to use
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
    def __init__(self, model_name, dropout_rate=0.0):
        super(CustomDPRContextEncoder, self).__init__()
        self.model = DPRContextEncoder.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return pooled_output




class CustomDPRQuestionEncoderWithDropout(nn.Module):
    def __init__(self, model_name, dropout_rate=0.0):
        super(CustomDPRQuestionEncoderWithDropout, self).__init__()
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_dropout = nn.Dropout(dropout_rate)
        # Create a multi-head attention layer
        self.attention = MultiheadAttention(embed_dim=768, num_heads=32)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Create a mask for current_turn and history tokens
        current_turn_mask = token_type_ids == 1
        history_mask = token_type_ids == 0

        # Apply masks to the hidden_states
        hidden_states = outputs.hidden_states[-1] # Get the last hidden states
        current_turn_embeddings = hidden_states.masked_fill(~current_turn_mask.unsqueeze(-1), 0)
        history_embeddings = hidden_states.masked_fill(~history_mask.unsqueeze(-1), 0)

        # Perform average pooling for both current_turn and history embeddings
        epsilon = 1e-8
        current_turn_avg_pool = current_turn_embeddings.sum(1) / (current_turn_mask.sum(1, keepdim=True) + epsilon)
        history_avg_pool = history_embeddings.sum(1) / (history_mask.sum(1, keepdim=True) + epsilon)

        # Stack the three tensors along the sequence dimension
        stacked_tensors = torch.stack((pooled_output, current_turn_avg_pool, history_avg_pool), dim=1)

        # Apply the multi-head attention to the stacked tensors (requires input and output shapes: [sequence_length, batch_size, embed_dim])
        attention_output, _ = self.attention(stacked_tensors.transpose(0, 1), stacked_tensors.transpose(0, 1), stacked_tensors.transpose(0, 1))
        attention_output = self.attention_dropout(attention_output)
        # Transpose back to the original shape [batch_size, sequence_length, embed_dim]
        attention_output = attention_output.transpose(0, 1)

        # Reduce the sequence dimension (e.g., by averaging or concatenating)
        mean_pooled_output  = attention_output.mean(dim=1)
        max_pooled_output = attention_output.max(dim=1).values
        final_output = (mean_pooled_output + max_pooled_output) / 2

        return final_output 

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


checkpoint_path_dpr = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/dpr-att/dpr_checkpoint.pth"
def load_saved_model(checkpoint_path_dpr):
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

    checkpoint_dpr = torch.load(checkpoint_path_dpr, map_location=device)
    combined_model.load_state_dict(checkpoint_dpr['model_state_dict'], strict=False)
    combined_model.to(device)
    combined_model.eval()

    return combined_model


# combined_model, t5_cross_encoder = load_saved_model(checkpoint_path_dpr, checkpoint_path_ce)
combined_model = load_saved_model(checkpoint_path_dpr)
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

def encode_questions(train_dataset, batch_size=16):
    encoded_questions = []
    max_length = 512
    dataset_length = len(train_dataset)
    
    for start_idx in tqdm(range(0, dataset_length, batch_size), desc="Encoding questions"):
        end_idx = min(start_idx + batch_size, dataset_length)
        questions = [train_dataset[idx]["question"] for idx in range(start_idx, end_idx)]
        
        encodings = question_tokenizer(questions, return_tensors='pt', padding=True, truncation=True, max_length=max_length, return_token_type_ids=True)
        input_ids = encodings["input_ids"].to(device)
        attention_masks = encodings["attention_mask"].to(device)
        token_type_ids = encodings["token_type_ids"].to(device)
        
        with torch.no_grad():
            outputs = combined_model.question_encoder(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)
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

def encode_context(corpus_dataset, batch_size=16):
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

def search_faiss(encoded_questions, index, k=10):
    D, I = index.search(encoded_questions, k)
    return I


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

# Calculate recall@1, recall@5, recall@10 scores
recall_1 = recall_at_k(top_10_indices, positive_psg_ids, 1)
recall_5 = recall_at_k(top_10_indices, positive_psg_ids, 5)
recall_10 = recall_at_k(top_10_indices, positive_psg_ids, 10)

print(f"Recall@1: {recall_1:.4f}")
print(f"Recall@5: {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")




from sklearn.metrics.pairwise import cosine_similarity

def search_cosine_similarity(encoded_questions, context_embeddings, k=10):
    similarity_matrix = cosine_similarity(encoded_questions, context_embeddings)
    I = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]
    return I

# Search using cosine similarity to find the top 10 most similar context embeddings
top_10_indices_cosine = search_cosine_similarity(question_embeddings, context_embeddings, k=10)

# Calculate recall@1, recall@5, recall@10 scores
recall_1_cosine = recall_at_k(top_10_indices_cosine, positive_psg_ids, 1)
recall_5_cosine = recall_at_k(top_10_indices_cosine, positive_psg_ids, 5)
recall_10_cosine = recall_at_k(top_10_indices_cosine, positive_psg_ids, 10)

print(f"Recall@1 (Cosine): {recall_1_cosine:.4f}")
print(f"Recall@5 (Cosine): {recall_5_cosine:.4f}")
print(f"Recall@10 (Cosine): {recall_10_cosine:.4f}")
