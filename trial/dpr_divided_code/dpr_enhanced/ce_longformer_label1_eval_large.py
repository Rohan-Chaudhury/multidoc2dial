
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import concurrent.futures
import math
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



  # Set this to the index of the GPU you want to use
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

# with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.validation.json", "r") as f:
#     validation_data = json.load(f)

# with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.test.json", "r") as f:
#     test_data = json.load(f)


with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.validation.json", "r") as f:
    test_data = json.load(f)

class CustomDPRContextEncoder(nn.Module):
    def __init__(self, model_name):
        super(CustomDPRContextEncoder, self).__init__()
        self.model = DPRContextEncoder.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        # print (linear_output.shape)
        # print (pooled_output.shape)
        return pooled_output




class CustomDPRQuestionEncoderWithDropout(nn.Module):
    def __init__(self, model_name):
        super(CustomDPRQuestionEncoderWithDropout, self).__init__()
        self.model = DPRQuestionEncoder.from_pretrained(model_name)

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




checkpoint_path_dpr = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/best_model_state.bin"


print ("Checkpoint path: ", checkpoint_path_dpr)


def load_saved_model(checkpoint_path_dpr):
    question_encoder = CustomDPRQuestionEncoderWithDropout("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
    context_encoder = CustomDPRContextEncoder(model_name="sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")

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
    return combined_model



combined_model = load_saved_model(checkpoint_path_dpr)
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

def encode_questions(train_dataset, batch_size=128):
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
def encode_context(corpus_dataset, batch_size=128):
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

del combined_model
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

from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig

# Search the FAISS index to find the top 10 most similar context embeddings
top_10_indices = search_faiss(question_embeddings, index, k=50)

# cross_encoder_path = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/longformer_models/margin_loss_large_19/checkpoint-5983"
cross_encoder_path = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/longformer_models/margin_loss_large_19/checkpoint-5983"
print ("Cross encoder path: ", cross_encoder_path)


config = LongformerConfig.from_pretrained(cross_encoder_path, num_labels=1)
config.gradient_checkpointing = False
config.attention_window = 256
# config.attention_probs_dropout_prob = 0.2
# config.hidden_dropout_prob = 0.2


cross_model = LongformerForSequenceClassification.from_pretrained(cross_encoder_path,config=config)
cross_model= nn.DataParallel(cross_model)
cross_model.to(device)

cross_tokenizer = LongformerTokenizerFast.from_pretrained(cross_encoder_path, max_length = 1024)



# def rank_sentences(question, candidate_sentences, model, tokenizer):
#     """
#     Rank the given candidate sentences based on their relevance to the given question.
    
#     Parameters:
#     - question: The input question string.
#     - candidate_sentences: A list of sentences (strings) that need to be ranked.
#     - model: The trained Longformer model.
#     - tokenizer: The tokenizer used during training.

#     Returns:
#     - A list of tuples, where each tuple is (sentence, score), sorted by descending scores.
#     """
    
#     # Model to evaluation mode
#     model.eval()

#     # Prepare the input data
#     input_data = []
#     for sentence in candidate_sentences:
#         text = f"[QUESTION] {question} </s> [CONTEXT] {sentence}"
#         input_data.append(text)

#     # Tokenize input data
#     inputs = tokenizer(input_data, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    
#     # Add global attention mask
#     global_attention_masks = [create_global_attention_mask(text, tokenizer) for text in input_data]
#     inputs["global_attention_mask"] = torch.tensor(global_attention_masks)

#     # Move input to appropriate device
#     inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

#     # Pass through the model
#     with torch.no_grad():
#         outputs = model(**inputs)
#         scores = outputs.logits[:, 1]  # We extract the score corresponding to the label 1

#     # Sort the sentences based on the scores
#     sorted_indices = scores.argsort(descending=True).tolist()

#     ranked_sentences = [(candidate_sentences[i], scores[i].item()) for i in sorted_indices]

#     return ranked_sentences


# # Example usage:
# question = "What is the capital of France?"
# candidate_sentences = ["Paris is a beautiful city.", "France is in Europe.", "The Eiffel Tower is in Paris.", "French cuisine is famous worldwide."]
# ranked_results = rank_sentences(question, candidate_sentences, model, tokenizer)

# for sentence, score in ranked_results:
#     print(f"Score: {score:.4f} - Sentence: {sentence}")





import math
from tqdm import tqdm

def rerank_with_cross_encoder(cross_questions, top_10_indices, context_passages, cross_model, cross_tokenizer, k=10, batch_size=1):
    cross_model.eval()
    max_length = 1024
    num_batches = math.ceil(len(cross_questions) / batch_size)
    reranked_indices = []

    for batch_idx in tqdm(range(num_batches)):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(cross_questions))

        batch_questions = cross_questions[batch_start:batch_end]
        batch_top_indices = top_10_indices[batch_start:batch_end]

        cross_input = [f"[QUESTION] {question} </s> [CONTEXT] {context_passages[top_idx]}" for question, top_indices in zip(batch_questions, batch_top_indices) for top_idx in top_indices]
        
        global_attention_masks = torch.tensor([create_global_attention_mask(text, cross_tokenizer) for text in cross_input])
        tokenized_inputs = cross_tokenizer(cross_input, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}
    
        global_attention_masks = global_attention_masks.to(device)

        with torch.no_grad():
            logits = cross_model(**tokenized_inputs, 
                       global_attention_mask=global_attention_masks).logits.squeeze().detach().cpu().numpy()
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


import numpy as np

import numpy as np


def search_cosine_similarity(encoded_questions, context_embeddings, k=50):
    similarity_matrix = cosine_similarity(encoded_questions, context_embeddings)
    I = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :k]
    return I

print("----50 50 100------")
# Search using cosine similarity to find the top 10 most similar context embeddings
top_10_indices_cosine = search_cosine_similarity(question_embeddings, context_embeddings, k=100)
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
MAX_SEQ_LEN = 1024
def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def truncate_question_sequences(question, tokenizer, max_question_len=500):
    """
    Truncates a given question to a maximum length of tokens using the provided tokenizer.
    Returns the truncated text.
    """
    tokenized_question = tokenizer(question, truncation=True, max_length=max_question_len, return_tensors="pt")
    # print ("Length of tokenized question is: ", len(tokenized_question["input_ids"].squeeze()))
    truncated_text = tokenizer.decode(tokenized_question["input_ids"].squeeze(), skip_special_tokens=True)
    return truncated_text

def preprocess_question(question):
    question = truncate_question_sequences(question, cross_tokenizer, max_question_len=256)
    question = question.replace("[SEP]", " [HISTORY] ")
    question = remove_extra_spaces(question)
    return question

def create_global_attention_mask(input_text, tokenizer, max_len=MAX_SEQ_LEN):

    tokens = tokenizer.tokenize(input_text)
    attention_mask = [0] * max_len  # initialize with zeros

    apply_attention = True
    for i, token in enumerate(tokens):

        if token == "</s>":
            apply_attention = False
            # print ("token is: ", i)

        if apply_attention:
            attention_mask[i] = 1

    # print ("attention mask is: ", sum (attention_mask))
    return attention_mask

cross_questions = [preprocess_question(question) for question in train_dataset["question"]]



reranked_indices_cosine = rerank_with_cross_encoder(cross_questions, top_10_indices_cosine, corpus_data_list , cross_model, cross_tokenizer, k=10)
# Calculate recall@1, recall@5, recall@10 scores
recall_1_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 1)
recall_5_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 5)
recall_10_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 10)

print(f"Recall@1 (Cosine): {recall_1_cosine:.4f}")
print(f"Recall@5 (Cosine): {recall_5_cosine:.4f}")
print(f"Recall@10 (Cosine): {recall_10_cosine:.4f}")


# reranked_indices_cosine = rerank_with_cross_encoder(cross_questions, reranked_indices_cosine, corpus_data_list , cross_model, cross_tokenizer, k=10)
# # Calculate recall@1, recall@5, recall@10 scores
# recall_1_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 1)
# recall_5_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 5)
# recall_10_cosine = recall_at_k(reranked_indices_cosine, positive_psg_ids, 10)

# print(f"Recall@1 (Cosine): {recall_1_cosine:.4f}")
# print(f"Recall@5 (Cosine): {recall_5_cosine:.4f}")
# print(f"Recall@10 (Cosine): {recall_10_cosine:.4f}")

