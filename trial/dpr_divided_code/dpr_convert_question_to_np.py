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



# checkpoint_path_dpr = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/2023-04-28_23-47-54_dpr/dpr_checkpoint.pth"
checkpoint_path_dpr = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/66_test/dpr_checkpoint.pth"
# checkpoint_path_ce =  "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/t5_large_500/2023-05-03_12-06-33/ce_checkpoint.pth"

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

    # checkpoint_ce = torch.load(checkpoint_path_ce, map_location=device)
    # t5_pretrained_model_name = "t5-large"
    # t5_cross_encoder = T5CrossEncoder(t5_pretrained_model_name, model_max_length)
    # t5_cross_encoder= nn.DataParallel(t5_cross_encoder)
    # t5_cross_encoder.to(device)
    # t5_cross_encoder.load_state_dict(checkpoint_ce['cross_encoder_state_dict'])
    # t5_cross_encoder.eval()
    # return combined_model, t5_cross_encoder
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



# def encode_context(corpus_dataset):
#     encoded_contexts = []
#     max_length = 512
#     for idx in range(len(corpus_dataset)):
#         context = corpus_dataset[idx]["text"]
#         encodings = context_tokenizer(context, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
#         input_ids = encodings["input_ids"].to(device)
#         attention_masks= encodings["attention_mask"].to(device)
#         with torch.no_grad():
#            embeddings = combined_model.context_encoder(input_ids, attention_masks).detach().cpu().numpy()
#         encoded_contexts.append(embeddings)
    
#     return np.vstack(encoded_contexts)

# # Assuming corpus_data_dict is your preprocessed data
# corpus_dataset = Dataset.from_dict(corpus_data_dict)

# # Encode the preprocessed corpus data
# context_embeddings = encode_context(corpus_dataset)

# # Save context embeddings offline
# np.save("context_embeddings.npy", context_embeddings)


from tqdm import tqdm

def encode_questions(train_dataset, batch_size=16):
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

print("Length of embeddings:", len(question_embeddings))
# Save context embeddings offline
np.save("question_embeddings.npy", question_embeddings)

