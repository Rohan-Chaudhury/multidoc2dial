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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set this to the index of the GPU you want to use
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





# checkpoint_path_dpr = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/2023-04-28_23-47-54_dpr/dpr_checkpoint.pth"
# checkpoint_path_dpr = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/66_test/dpr_checkpoint.pth"
checkpoint_path_ce =  "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/t5_large_500/2023-05-03_12-06-33/ce_checkpoint.pth"

def load_saved_model(checkpoint_path_ce):
    # question_encoder = CustomDPRQuestionEncoderWithDropout("sivasankalpp/dpr-multidoc2dial-structure-question-encoder", 0.0)
    # context_encoder = CustomDPRContextEncoder(model_name="sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder", dropout_rate=0.0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # question_encoder = nn.DataParallel(question_encoder)
    # context_encoder = nn.DataParallel(context_encoder)

    # question_encoder.to(device)
    # question_encoder.eval()
    # context_encoder.to(device)
    # context_encoder.eval()
    # combined_model = DPRCombinedModel(question_encoder, context_encoder)

    # checkpoint_dpr = torch.load(checkpoint_path_dpr, map_location=device)
    # combined_model.load_state_dict(checkpoint_dpr['model_state_dict'], strict=False)
    # combined_model.to(device)
    # combined_model.eval()
    model_max_length = 1024
    checkpoint_ce = torch.load(checkpoint_path_ce, map_location=device)
    t5_pretrained_model_name = "t5-large"
    t5_cross_encoder = T5CrossEncoder(t5_pretrained_model_name, model_max_length)
    t5_cross_encoder= nn.DataParallel(t5_cross_encoder)
    t5_cross_encoder.to(device)
    t5_cross_encoder.load_state_dict(checkpoint_ce['cross_encoder_state_dict'])
    t5_cross_encoder.eval()
    return combined_model, t5_cross_encoder
    return combined_model


# combined_model, t5_cross_encoder = load_saved_model(checkpoint_path_dpr, checkpoint_path_ce)
combined_model = load_saved_model(checkpoint_path_dpr)
print ("Model Loaded")

context_embeddings = np.load("context_embeddings.npy")

question_embeddings = np.load("question_embeddings.npy")

print ("No.of questions: ",len(question_embeddings))
index = faiss.IndexFlatIP(context_embeddings.shape[1])

index.add(context_embeddings)

print(f"Number of context embeddings in the FAISS index: {index.ntotal}")


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
