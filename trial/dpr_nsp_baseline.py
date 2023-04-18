
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
from transformers import DPRConfig, DPRContextEncoder, DPRContextEncoderTokenizer, AutoTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRReader, DPRReaderTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from tqdm.auto import tqdm
from transformers import T5EncoderModel
from transformers import AutoModel


config = {
    "epochs": 10,
    "batch_size": 4,
    "learning_rates": {
        "question_encoder": 3e-5,
        "context_encoder": 3e-5,
        "cross_encoder": 1e-5
    },
    "gradient_accumulation_steps": 128,
    "max_length": 512,
    "patience": 3,
    "temperature": 1.0,
}



question_tokenizer = AutoTokenizer.from_pretrained(
    "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
context_tokenizer = AutoTokenizer.from_pretrained(
    "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")

class T5CrossEncoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.t5.config.d_model, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits




class CustomDPRQuestionEncoderWithDropout(nn.Module):
    def __init__(self, model_name, dropout_rate):
        super(CustomDPRQuestionEncoderWithDropout, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return  self.linear(self.dropout(self.layer_norm(pooled_output)))
        # return self.linear(pooled_output)
        # return outputs.pooler_output

def compute_scores(question_embeddings, context_embeddings):
    return torch.matmul(question_embeddings, context_embeddings.t())

class CustomDPRContextEncoder(nn.Module):
    def __init__(self, model_name, dropout_rate):
        super(CustomDPRContextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return  self.linear(self.dropout(self.layer_norm(pooled_output)))
        # return self.linear(pooled_output)
        # return outputs.pooler_output

# config = DPRConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# question_encoder = CustomDPRQuestionEncoderWithDropout("facebook/dpr-question_encoder-single-nq-base", 0.1)


# reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
# reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set this to the index of the GPU you want to use
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.train.json", "r") as f:
    training_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.psg.multidoc2dial_all.structure.json", "r") as f:
    corpus_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.validation.json", "r") as f:
    validation_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.test.json", "r") as f:
    test_data = json.load(f)


import json
import re
from datasets import Dataset

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_question(question):


    return question




import random



def preprocess_corpus_data(corpus_data):
    corpus_data_preprocessed = {
        "title": [],
        "text": []
    }

    for item in corpus_data:
        title = remove_extra_spaces(item["title"].lower())
        text = remove_extra_spaces(item["text"].lower())
        corpus_data_preprocessed["title"].append(title)
        corpus_data_preprocessed["text"].append(text)
    
    return corpus_data_preprocessed

# corpus_data_dict = preprocess_corpus_data(corpus_data)

# corpus_data_dict = preprocess_corpus_data(corpus_data)
# corpus_dataset = Dataset.from_dict(corpus_data_dict)



# # Assuming `training_data` is a list of dictionaries containing "question", "positive_ctxs", "negative_ctxs", and "hard_negative_ctxs"
# preprocessed_data = preprocess_data(training_data)
# train_dataset = Dataset.from_dict(preprocessed_data)
# train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

# preprocessed_validation_data = preprocess_data(validation_data)
# validation_dataset  = Dataset.from_dict(preprocessed_validation_data)
# validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True)


# Add the preprocessing function that tokenizes the context and questio


def contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=1.0):
    pos_distances = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
    neg_distances = torch.norm(anchor_embeddings - negative_embeddings, dim=-1)
    return F.relu(pos_distances - neg_distances + margin).mean()

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



def load_saved_model():
    question_encoder = DPRQuestionEncoder.from_pretrained(
        "sivasankalpp/dpr-multidoc2dial-structure-question-encoder")


    context_encoder = DPRContextEncoder.from_pretrained(
        "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    question_encoder = nn.DataParallel(question_encoder)
    context_encoder = nn.DataParallel(context_encoder)

    question_encoder.to(device)
    context_encoder.to(device)
    question_encoder.eval()
    context_encoder.eval()
    return question_encoder, context_encoder

question_encoder, context_encoder = load_saved_model()


import torch
import torch.nn as nn

def cosine_similarity_torch(embedding1, embedding2):
    # Ensure input tensors are in the right shape (batch_size x embedding_size)
    embedding1 = embedding1.unsqueeze(0) if len(embedding1.shape) == 1 else embedding1
    embedding2 = embedding2.unsqueeze(0) if len(embedding2.shape) == 1 else embedding2
    
    # Create a CosineSimilarity instance
    cosine_sim = nn.CosineSimilarity(dim=1)

    # Calculate cosine similarity between the two embeddings
    similarity_scores = cosine_sim(embedding1, embedding2)

    return similarity_scores


def evaluate(question_encoder, context_encoder, question_tokenizer, context_tokenizer, data_loader, device):
    # model.eval()
    question_encoder.eval()
    context_encoder.eval()
    f1_scores = []
    mrr_scores = []
    r_at_1 = []
    r_at_5 = []
    r_at_10 = []
    em_scores = []
    max_length = 512
    ik = 0
    print(len(data_loader))
    for idx,item in enumerate(tqdm(data_loader)):
        with torch.no_grad():
        # with tqdm(data_loader, desc="Evaluating") as pbar:
           
                # print(batch)
                # question_inputs = {k: v.to(device) for k, v in batch["question_inputs"].items()}
                # positive_ctx_inputs = {k: v.to(device) for k, v in batch["positive_ctx_inputs"].items()}
                # negative_ctx_inputs_list = batch["negative_ctx_inputs_list"]
                # question_outputs, positive_ctx_outputs = model(
                #     question_inputs["input_ids"], question_inputs["attention_mask"],
                #     positive_ctx_inputs["input_ids"], positive_ctx_inputs["attention_mask"]
                # )
                

                negative_contexts= []
                question = preprocess_question(item["question"])
                positive_ctxs = item["positive_ctxs"]
                negative_ctxs = item["negative_ctxs"]
                hard_negative_ctxs = item["hard_negative_ctxs"]

                for positive_ctx in positive_ctxs:
                    positive_context = remove_extra_spaces(positive_ctx["title"] + " " + positive_ctx["text"])

                # Combine negative_ctxs and hard_negative_ctxs for sampling
                    all_negative_ctxs = (negative_ctxs * 1) + (hard_negative_ctxs * 2)

                for negative_ctx in all_negative_ctxs:
                    negative_context = remove_extra_spaces(negative_ctx["title"] + " " + negative_ctx["text"])

                    negative_contexts.append(negative_context)

                # print(len(negative_contexts))
                # print(len(positive_context[0]))
                anchor_encodings = question_tokenizer(question, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                positive_encodings = context_tokenizer(positive_context, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
                negative_encodings = context_tokenizer(negative_contexts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

                # print(positive_encodings)
                # print(negative_encodings)
                anchor_input_ids, anchor_attention_mask = anchor_encodings['input_ids'].to(device), anchor_encodings['attention_mask'].to(device)
                positive_input_ids, positive_attention_mask = positive_encodings['input_ids'].to(device), positive_encodings['attention_mask'].to(device)
                # negative_input_ids, negative_attention_mask = negative_encodings['input_ids'].to(device), negative_encodings['attention_mask'].to(device)


                # Compute embeddings
                anchor_embeddings = question_encoder(anchor_input_ids, anchor_attention_mask).pooler_output
                positive_embeddings = context_encoder(positive_input_ids, positive_attention_mask).pooler_output

                negative_input_ids = negative_encodings["input_ids"].to(device)
                negative_attention_masks= negative_encodings["attention_mask"].to(device)
                negative_embeddings = context_encoder(negative_input_ids, negative_attention_masks).pooler_output
                # for i in range(len(negative_input_ids)):
                #     print(i)
                #     negative_embeddings.append())
                # negative_embeddings = [model.context_encoder(input_ids.to(device), attention_mask.to(device)) for input_ids,attention_mask in zip(negative_encodings["input_ids"], negative_encodings["attention_mask"])]
                # print("\n\n\n\n\n\n\n")
                # print(len(negative_embeddings))
                # print(len(negative_embeddings[0]))
                # print(len(positive_embeddings))
                # print(len(positive_embeddings[0]))
                # print("\n\n\n\n\n\n\n")

                similarity_scores = [cosine_similarity_torch(anchor_embeddings, positive_embeddings).item()]

                embedding_to_passage = {str(positive_embeddings): positive_context}
                for negative_context, negative_embedding in zip(negative_contexts, negative_embeddings):
                    embedding_to_passage[str(negative_embedding)] = negative_context


                for negative_embedding in negative_embeddings:
                    # for negative_embedding in b_negative_embedding:
                        similarity_scores.append(cosine_similarity_torch(anchor_embeddings, negative_embedding).item())

                # print(similarity_scores)
                # Sort indices according to their similarity scores
                ranked_indices = np.argsort(similarity_scores)[::-1]
                # print(ranked_indices)
                # Select the top k embeddings based on their ranked indices
                top_k_embeddings = [positive_embeddings if i == 0 else negative_embeddings[i-1] for i in ranked_indices[:20]]
                all_embeddings = [positive_embeddings if i == 0 else negative_embeddings[i-1] for i in ranked_indices]
                top_k_passages = [embedding_to_passage[str(embedding)] for embedding in top_k_embeddings]
                all_passages = [embedding_to_passage[str(embedding)] for embedding in all_embeddings]
                
                # ranks = rankdata(similarity_scores, method="max")
                # positive_ctx_rank = ranks[0]
                # print("\n\n")
                # print(len(similarity_scores), positive_ctx_rank, ranks, ranked_indices)
                # print("\n\n")


                # Re-rank the top-k passages using the cross-encoder
                # print (reranked_indices)

                new_positive_ctx_rank = -1
                for i, passage in enumerate(top_k_passages):
                    # print("\n\n\n\n\n\n")
                    # print(passage)
                    # print("\n\n\n\n\n\n")
                    # print(positive_context)
                    if passage == positive_context:
                        new_positive_ctx_rank = i 
                        # print("new_positive_ctx_rank", new_positive_ctx_rank)
                        break
                if new_positive_ctx_rank == -1:
                    ik=ik+1
                    print(ik)
                    for i, passage in enumerate(all_passages):
                        if passage == positive_context:
                            new_positive_ctx_rank = i 
                            # print("new_positive_ctx_rank", new_positive_ctx_rank)
                            break
                binary_true = [1] + [0]  * (len(all_passages) - 1)
                binary_pred = [1 if rank == new_positive_ctx_rank else 0 for rank in range(len(binary_true))]
                f1_scores.append(f1_score(binary_true, binary_pred))
                
                mrr_scores.append(1 / (new_positive_ctx_rank+1))
                r_at_1.append(1 if new_positive_ctx_rank == 0 else 0)
                r_at_5.append(1 if new_positive_ctx_rank < 5 else 0)
                r_at_10.append(1 if new_positive_ctx_rank < 10 else 0)

                # Calculate EM (Exact Match) score
                em_scores.append(1 if new_positive_ctx_rank == 0 else 0)
                # pbar.set_postfix({"F1": np.mean(f1_scores), "MRR": np.mean(mrr_scores), "R@1": np.mean(r_at_1), "R@5": np.mean(r_at_5), "R@10": np.mean(r_at_10), "EM": np.mean(em_scores)})
                # print("done")
                if idx%100 == 0:
                    print(idx)
                    print("F1", np.mean(f1_scores))
                    print("MRR", np.mean(mrr_scores))
                    print("R@1", np.mean(r_at_1))
                    print("R@5", np.mean(r_at_5))
                    print("R@10", np.mean(r_at_10))
                    print("EM", np.mean(em_scores))
                    print(ik)
                    


    f1_avg = np.mean(f1_scores)
    mrr_avg = np.mean(mrr_scores)
    r_at_1_avg = np.mean(r_at_1)
    r_at_5_avg = np.mean(r_at_5)
    r_at_10_avg = np.mean(r_at_10)
    em_avg = np.mean(em_scores)

    return {
        "f1": f1_avg,
        "mrr": mrr_avg,
        "r@1": r_at_1_avg,
        "r@5": r_at_5_avg,
        "r@10": r_at_10_avg,
        "em": em_avg
    }




with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.test.json", "r") as f:
    test_data = json.load(f)



test_data= test_data

evaluation_results = evaluate(question_encoder, context_encoder, question_tokenizer, context_tokenizer, test_data, device)

print("Evaluation results:")
for metric, value in evaluation_results.items():
    print(f"{metric}: {value}")

