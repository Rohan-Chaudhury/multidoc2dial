import json
import torch
from transformers import (
    DPRContextEncoder, 
    DPRQuestionEncoder
    
)
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import transformers

print(transformers.__version__)

import json
from torch.optim.lr_scheduler import CyclicLR
from transformers import DPRConfig, DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from tqdm.auto import tqdm

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

config = {
    "epochs": 100,
    "batch_size": 8,
    "learning_rates": {
        "question_encoder": 1e-5,
        "context_encoder": 1e-5,
    },
    "gradient_accumulation_steps":512,
    "max_length": 512,
    "patience": 40,
    "temperature": 1.0,
}

def kaiming_initialization(module, nonlinearity="leaky_relu", mode="fan_in"):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        # Set the negative slope for leaky_relu that approximates gelu
        negative_slope = 0.01 if nonlinearity == "leaky_relu" else 0
        nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity, a=negative_slope)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class CustomDPRContextEncoder(nn.Module):
    def __init__(self, model_name, dropout_rate):
        super(CustomDPRContextEncoder, self).__init__()
        self.model = DPRContextEncoder.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.linear = self._init_linear(self.model.config.hidden_size)

    def _init_linear(self, hidden_size):
        linear = nn.Linear(hidden_size, hidden_size)
        kaiming_initialization(linear, nonlinearity="leaky_relu")
        return linear

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        linear_output = self.linear(self.dropout(self.layer_norm(pooled_output)))
        # print (linear_output.shape)
        # print (pooled_output.shape)
        return linear_output + pooled_output




class CustomDPRQuestionEncoderWithDropout(nn.Module):
    def __init__(self, model_name, dropout_rate):
        super(CustomDPRQuestionEncoderWithDropout, self).__init__()
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.linear = self._init_linear(self.model.config.hidden_size)

    def _init_linear(self, hidden_size):
        linear = nn.Linear(hidden_size, hidden_size)
        kaiming_initialization(linear, nonlinearity="leaky_relu")
        return linear

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        linear_output = self.linear(self.dropout(self.layer_norm(pooled_output)))
        return linear_output + pooled_output


def compute_scores(question_embeddings, context_embeddings):
    return cosine_similarity(question_embeddings, context_embeddings, dim=-1)

# config = DPRConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = CustomDPRQuestionEncoderWithDropout("sivasankalpp/dpr-multidoc2dial-structure-question-encoder", 0.1)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")

context_encoder = CustomDPRContextEncoder(model_name="sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder", dropout_rate=0.1)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")


# reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
# reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_encoder = nn.DataParallel(question_encoder)
context_encoder = nn.DataParallel(context_encoder)
question_encoder.to(device)
context_encoder.to(device)





with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.train.json", "r") as f:
    training_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.psg.multidoc2dial_all.structure.json", "r") as f:
    corpus_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.validation.json", "r") as f:
    validation_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.test.json", "r") as f:
    test_data = json.load(f)



def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_question(question):
    return remove_extra_spaces(question)


def preprocess_data(training_data, negative_weight=1, hard_negative_weight=2):
    train_data = {
        "question": [],
        "positive_context": [],
        "negative_context": []
    }

    for item in training_data:
        question = preprocess_question(item["question"])
        positive_ctxs = item["positive_ctxs"]
        negative_ctxs = item["negative_ctxs"]
        hard_negative_ctxs = item["hard_negative_ctxs"]

        for positive_ctx in positive_ctxs:
            positive_context = remove_extra_spaces(positive_ctx["text"])

            # Combine negative_ctxs and hard_negative_ctxs for sampling
            all_negative_ctxs = (negative_ctxs * negative_weight) + (hard_negative_ctxs * hard_negative_weight)

            for negative_ctx in all_negative_ctxs:
                negative_context = remove_extra_spaces(negative_ctx["text"])

                train_data["question"].append(question)
                train_data["positive_context"].append(positive_context)
                train_data["negative_context"].append(negative_context)
                

    return train_data

def preprocess_corpus_data(corpus_data):
    corpus_data_preprocessed = {
        "title": [],
        "text": []
    }

    for item in corpus_data:
        title = remove_extra_spaces(item["title"])
        text = remove_extra_spaces(item["text"])
        corpus_data_preprocessed["title"].append(title)
        corpus_data_preprocessed["text"].append(text)
    
    return corpus_data_preprocessed

# corpus_data_dict = preprocess_corpus_data(corpus_data)

corpus_data_dict = preprocess_corpus_data(corpus_data)
corpus_dataset = Dataset.from_dict(corpus_data_dict)


# Assuming `training_data` is a list of dictionaries containing "question", "positive_ctxs", "negative_ctxs", and "hard_negative_ctxs"
preprocessed_data = preprocess_data(training_data)
train_dataset = Dataset.from_dict(preprocessed_data)
train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

preprocessed_validation_data = preprocess_data(validation_data)
validation_dataset  = Dataset.from_dict(preprocessed_validation_data)
validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True)



import torch

from transformers import DPRQuestionEncoder, DPRContextEncoder
from torch.nn.utils import clip_grad_norm_



class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif self.best_loss - loss > self.delta:
            self.counter = 0
            self.best_loss = loss
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        return False
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
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from math import ceil
from tqdm import tqdm
import copy
import random
import datetime
import os


def process_batch(batch, question_tokenizer, context_tokenizer, max_length, device):
    questions = batch["question"]
    positive_contexts = batch["positive_context"]
    negative_contexts = batch["negative_context"]

    anchor_encodings = question_tokenizer(questions, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    positive_encodings = context_tokenizer(positive_contexts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    negative_encodings = context_tokenizer(negative_contexts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

    anchor_input_ids, anchor_attention_mask = anchor_encodings['input_ids'].to(device), anchor_encodings['attention_mask'].to(device)
    positive_input_ids, positive_attention_mask = positive_encodings['input_ids'].to(device), positive_encodings['attention_mask'].to(device)
    negative_input_ids, negative_attention_mask = negative_encodings['input_ids'].to(device), negative_encodings['attention_mask'].to(device)

    return anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask



def train_dpr_model(config, train_dataset, validation_dataset, model, question_tokenizer, context_tokenizer,device):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rates = config["learning_rates"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]

    max_length = config["max_length"]
    patience = config["patience"]


    weight_decay = 1e-5

    adam_epsilon = 1e-8
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_question_encoder_parameters = [
        {
            "params": [p for n, p in model.question_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.question_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer_grouped_context_encoder_parameters = [
        {
            "params": [p for n, p in model.context_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.context_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    early_stopping = EarlyStopping(patience=patience)
    
    question_encoder_optimizer = AdamW(optimizer_grouped_question_encoder_parameters , lr=learning_rates['question_encoder'],  eps=adam_epsilon)
    context_encoder_optimizer = AdamW(optimizer_grouped_context_encoder_parameters, lr=learning_rates['context_encoder'],  eps=adam_epsilon)
    number_of_batches = ceil(len(train_dataset) / batch_size)
    total_steps = number_of_batches * epochs // gradient_accumulation_steps

    base_learning_rate = 1e-5

    T_0 = 2000  # Number of iterations for the first restart
    T_mult = 2  # Multiplicative factor to increase the cycle length after each restart

    question_encoder_scheduler = CosineAnnealingWarmRestarts(question_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)

    context_encoder_scheduler = CosineAnnealingWarmRestarts(context_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)


    # model.to(device)
    best_val_loss = float('inf')
    best_model = None

    # Initialize loss weights
    contrastive_weight = 1 / 2

    dr_weight = 1 / 2

    # Initialize running averages for losses
    running_contrastive_loss = 0.0
  
    running_dr_loss = 0.0

    # Define the smoothing factor for updating the running averages
    smoothing_factor = 0.1

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0


        train_iter=tqdm(train_dataloader, desc="Training", ncols=100)
        for idx, batch in enumerate(train_iter):
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = process_batch(batch, question_tokenizer, context_tokenizer, max_length, device)
            # example_indices = shuffled_indices[i:i + batch_size]

            # Compute embeddings
            anchor_embeddings = model.question_encoder(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = model.context_encoder(positive_input_ids, positive_attention_mask)
            negative_embeddings = model.context_encoder(negative_input_ids, negative_attention_mask)

            # Compute contrastive loss
            loss = contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            positive_scores = compute_scores(anchor_embeddings, positive_embeddings)
            negative_scores = compute_scores(anchor_embeddings, negative_embeddings)
            margin = 0.1  # Adjust the margin value as needed based on the problem and dataset
            dr_loss = torch.clamp(margin - positive_scores + negative_scores, min=0).mean()



            # Update running averages for losses
            running_contrastive_loss = (1 - smoothing_factor) * running_contrastive_loss + smoothing_factor * loss.item()
            running_dr_loss = (1 - smoothing_factor) * running_dr_loss + smoothing_factor * dr_loss.item()

            # Update weights based on the running averages of losses
            total_running_loss = running_contrastive_loss + running_dr_loss
            contrastive_weight = running_contrastive_loss / total_running_loss
            dr_weight = running_dr_loss / total_running_loss

            # Compute the combined loss using updated weights
            combined_loss = contrastive_weight * loss + dr_weight * dr_loss
            total_loss += combined_loss.item()



        # Combine the losses
            # combined_loss = (1 - alpha) * loss + alpha * cross_encoder_loss + dr_loss
            # total_loss += combined_loss.item()

            # total_loss += loss.item()
            combined_loss_print = combined_loss.item()
            combined_loss = combined_loss / gradient_accumulation_steps
            combined_loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0 or (idx + 1 == len(train_dataloader)):
                # Apply gradient clipping
                clip_grad_norm_(model.question_encoder.parameters(), max_norm=1.0)
                clip_grad_norm_(model.context_encoder.parameters(), max_norm=1.0)


                question_encoder_optimizer.step()
                context_encoder_optimizer.step()


                question_encoder_scheduler.step()
                context_encoder_scheduler.step()


                question_encoder_optimizer.zero_grad()
                context_encoder_optimizer.zero_grad()
 

            
            train_iter.set_description(f"Training (loss = {combined_loss_print:.4f})")
            train_iter.refresh()

        avg_train_loss = total_loss / number_of_batches
        print(f"Training loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0.0
        number_of_batches_validation = ceil(len(validation_dataset) / batch_size)

        val_iter=tqdm(validation_dataloader, desc="Validation", ncols=100)
        for batch in val_iter:
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = process_batch(batch, question_tokenizer, context_tokenizer, max_length, device)

            with torch.no_grad():
                # Compute embeddings
                anchor_embeddings = model.question_encoder(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = model.context_encoder(positive_input_ids, positive_attention_mask)
                negative_embeddings = model.context_encoder(negative_input_ids, negative_attention_mask)
                positive_scores = compute_scores(anchor_embeddings, positive_embeddings)
                negative_scores = compute_scores(anchor_embeddings, negative_embeddings)
                
                margin = 1.0
                dr_loss = torch.clamp(margin - positive_scores + negative_scores, min=0).mean()

                # Compute contrastive loss
                loss = contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)





            # Combine the losses
            combined_loss =  loss + dr_loss


            total_val_loss += combined_loss.item()
            val_iter.set_description(f"Validation (loss = {combined_loss.item():.4f})")
            val_iter.refresh()
            
        avg_val_loss = total_val_loss / number_of_batches_validation
        print(f"Validation loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            # Get the current date and time as a string
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            directory_to_save="/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/contr_disr_ran_resid_pre/"+timestamp
            # Create a new directory with the timestamp
            os.makedirs(directory_to_save, exist_ok=True)

            # Save the model checkpoint with additional metadata
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state_dict,
                'question_encoder_optimizer_state_dict': question_encoder_optimizer.state_dict(),
                'context_encoder_optimizer_state_dict': context_encoder_optimizer.state_dict(),
                'question_encoder_scheduler_state_dict': question_encoder_scheduler.state_dict(),
                'context_encoder_scheduler_state_dict': context_encoder_scheduler.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(directory_to_save, "dpr_checkpoint.pth"))

            print(f"Best model checkpoint saved with validation loss: {best_val_loss} in directory {directory_to_save}")

            print(f"Best model saved with validation loss: {best_val_loss}")

        if early_stopping(avg_val_loss):
            print("Early stopping triggered")
            break

    print("Training complete.")
    return best_model



# Calculate total training steps and set num_warmup_steps as a fraction of total steps
total_steps = (len(train_dataset) // (config["batch_size"] * config["gradient_accumulation_steps"])) * config["epochs"]
config["num_warmup_steps"] = int(0.1 * total_steps)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


combined_model = DPRCombinedModel(question_encoder, context_encoder)
best_model = train_dpr_model(config, train_dataset, validation_dataset, combined_model, question_tokenizer, context_tokenizer, device)

