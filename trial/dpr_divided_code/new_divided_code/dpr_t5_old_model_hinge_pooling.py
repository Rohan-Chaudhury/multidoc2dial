import json
import torch

from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import transformers

print(transformers.__version__)
from transformers import T5Tokenizer
import json

from tqdm.auto import tqdm
from transformers import T5EncoderModel
import torch_optimizer as optim
from transformers import T5Config

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

config = {
    "epochs": 20,
    "batch_size": 2,
    "learning_rates": {
        "cross_encoder": 2e-5
    },
    "gradient_accumulation_steps": 2048,
    "max_length": 1024,
    "patience": 10,
    "temperature": 1.0,
}

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
        nn.init.kaiming_normal_(classifier.weight, nonlinearity='relu')
        return classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.layer_norm(pooled_output)
        logits = self.classifier(pooled_output)
        return logits





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_max_length = 1024

# checkpoint_path_ce =  "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/t5_large_500/2023-05-03_12-06-33/ce_checkpoint.pth"
# checkpoint_path_ce = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/output/models/66_ce/ce_checkpoint.pth"
checkpoint_path_ce = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/new_divided_code/output/t5_large_500/2023-05-21_12-41-12/ce_checkpoint.pth"

checkpoint_ce = torch.load(checkpoint_path_ce, map_location=device)

t5_pretrained_model_name = "t5-large"
t5_cross_encoder = T5CrossEncoder(t5_pretrained_model_name, model_max_length)
t5_cross_encoder= nn.DataParallel(t5_cross_encoder)
t5_cross_encoder.to(device)
t5_cross_encoder.load_state_dict(checkpoint_ce['cross_encoder_state_dict'])
t5_cross_encoder.train()
print ("\n Checkpoint loaded successfully! \n ")
print ("\n Loading checkpoint from: ", checkpoint_path_ce, "\n")
print("Loss: ", checkpoint_ce['loss'])
print ("Epoch: ", checkpoint_ce['epoch'])
del checkpoint_ce
print ("\n Model loaded successfully and checkpoint deleted! Now doing data cleaning! \n ")
t5_tokenizer = T5Tokenizer.from_pretrained(t5_pretrained_model_name, model_max_length=model_max_length)

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

def preprocess_data(training_data, negative_weight=1, hard_negative_weight=1):
    train_data = {
        "question": [],
        "positive_context": [],
        "negative_context": []
    }

    for item in tqdm(training_data):
        # print(item["question"])
        question = preprocess_question(item["question"])
        # print(question)
        # print("\nssdf\n")
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

preprocessed_data = preprocess_data(training_data, negative_weight=1, hard_negative_weight=1)


preprocessed_validation_data = preprocess_data(validation_data, negative_weight=1, hard_negative_weight=2)
print ("\n New Flan Datasets - whole full with new batch size with all hard negatives ----\n")

# preprocessed_data = {key: value[:100] for key, value in preprocessed_data.items()}
# preprocessed_validation_data = {key: value[:100] for key, value in preprocessed_validation_data.items()}

train_dataset = Dataset.from_dict(preprocessed_data)
train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
validation_dataset  = Dataset.from_dict(preprocessed_validation_data)
validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True)

print ("\n\n New Flan Datasets loaded with whole data\n\n")


import torch

from transformers import DPRQuestionEncoder, DPRContextEncoder
from torch.nn.utils import clip_grad_norm_



class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
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

    
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from math import ceil
from tqdm import tqdm
import copy
import random
import datetime
import os

from torch.nn.functional import relu




print ("Batch size is ", config["batch_size"])
print ("Gradient accumulation steps is ", config["gradient_accumulation_steps"])

def train_dpr_model(config, train_dataset, validation_dataset, cross_encoder, t5_tokenizer, device):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rates = config["learning_rates"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]


    # Initialize MarginRankingLoss
    margin_ranking_loss = torch.nn.MarginRankingLoss(margin=1.0)

    max_length = config["max_length"]
    patience = config["patience"]

    weight_decay = 1e-5  # Weight decay for regularization
    base_learning_rate = learning_rates['cross_encoder']  # Base learning rate for optimizer

    adam_epsilon = 1e-8  # Epsilon for numerical stability in AdamW optimizer
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_cross_encoder_parameters = [
        {
            "params": [p for n, p in cross_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in cross_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    early_stopping = EarlyStopping(patience=patience)

    # Initialize the optimizer
    cross_encoder_optimizer = AdamW(optimizer_grouped_cross_encoder_parameters, lr=base_learning_rate,eps=adam_epsilon)
        
    # cross_encoder_optimizer.load_state_dict(checkpoint_ce['cross_encoder_optimizer_state_dict'])
    cross_encoder_optimizer = optim.Lookahead(cross_encoder_optimizer)
    print ("\n----Cross Encoder Lookahead implemented with new scheduler----\n")
    number_of_batches = ceil(len(train_dataset) / batch_size)
    total_steps = number_of_batches * epochs // gradient_accumulation_steps

    eta_min = 1e-6  # Minimum learning rate during cosine annealing
    T_0 = 2000  # Number of iterations for the first restart
    T_mult = 2  # Multiplicative factor to increase the cycle length after each restart

    # Initialize the learning rate scheduler
    cross_encoder_scheduler = CosineAnnealingWarmRestarts(cross_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    # model.to(device)
    best_val_loss = float('inf')
    best_model = None




    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        train_iter=tqdm(train_dataloader, desc="Training", ncols=100)
        for idx, batch in enumerate(train_iter):

            questions = batch["question"]
            positive_contexts = batch["positive_context"]
            negative_contexts = batch["negative_context"]

            positive_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, positive_contexts)]
            negative_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, negative_contexts)]

            t5_encodings = t5_tokenizer(positive_t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
            t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)
            positive_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()

            t5_encodings = t5_tokenizer(negative_t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
            t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)
            negative_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()

            cross_encoder_labels = torch.tensor([1] * len(questions), dtype=torch.float, device=device)
            cross_encoder_loss = margin_ranking_loss(positive_logits, negative_logits, cross_encoder_labels)

            # Compute the combined loss using updated weights
            combined_loss =  cross_encoder_loss 
            total_loss += combined_loss.item()

            combined_loss = combined_loss / gradient_accumulation_steps
            combined_loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0  or (idx + 1 == len(train_dataloader)):
                # Apply gradient clipping
                clip_grad_norm_(cross_encoder.parameters(), max_norm=1.0)

                cross_encoder_optimizer.step()
                cross_encoder_scheduler.step()
                cross_encoder_optimizer.zero_grad()

            train_iter.set_description(f"Training (loss = {cross_encoder_loss.item():.4f})")
            train_iter.refresh()

        avg_train_loss = total_loss / number_of_batches
        print(f"Training loss: {avg_train_loss}")

        # Validation
        cross_encoder.eval()

        ###################################################
        # train_cross_encoder_state_dict = copy.deepcopy(cross_encoder.state_dict())
        # Get the current date and time as a string
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        directory_to_save_train="/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/new_divided_code/output/train/t5_large_500/"+timestamp
        # Create a new directory with the timestamp
        os.makedirs(directory_to_save_train, exist_ok=True)

        # Save the model checkpoint with additional metadata
        torch.save({
            'epoch': epoch,
            'cross_encoder_state_dict': cross_encoder.state_dict(),
            'cross_encoder_optimizer_state_dict': cross_encoder_optimizer.state_dict(),
            'loss': avg_train_loss,
        }, os.path.join(directory_to_save_train, "ce_checkpoint.pth"))

        print(f"model checkpoint saved with training loss: {avg_train_loss} in directory {directory_to_save_train}")

        #######################################
        # del train_cross_encoder_state_dict
        # torch.cuda.empty_cache()


        total_val_loss = 0.0
        number_of_batches_validation = ceil(len(validation_dataset) / batch_size)

        val_iter=tqdm(validation_dataloader, desc="Validation", ncols=100)
        for batch in val_iter:
            questions = batch["question"]
            positive_contexts = batch["positive_context"]
            negative_contexts = batch["negative_context"]

            positive_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, positive_contexts)]
            negative_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, negative_contexts)]

            with torch.no_grad():
                t5_encodings = t5_tokenizer(positive_t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
                t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)
                positive_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()

                t5_encodings = t5_tokenizer(negative_t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
                t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)
                negative_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()

                cross_encoder_labels = torch.tensor([1] * len(questions), dtype=torch.float, device=device)
                cross_encoder_loss = margin_ranking_loss(positive_logits, negative_logits, cross_encoder_labels)

            # Combine the losses
            combined_loss = cross_encoder_loss 

            total_val_loss += combined_loss.item()
            val_iter.set_description(f"Validation (loss = {cross_encoder_loss.item():.4f})")
            val_iter.refresh()
                    
        avg_val_loss = total_val_loss / number_of_batches_validation
        print(f"Validation loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # best_cross_encoder_state_dict = copy.deepcopy(cross_encoder.state_dict())
            # Get the current date and time as a string
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            directory_to_save="/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/new_divided_code/output/t5_large_500/"+timestamp
            # Create a new directory with the timestamp
            os.makedirs(directory_to_save, exist_ok=True)

            # Save the model checkpoint with additional metadata
            torch.save({
                'epoch': epoch,
                'cross_encoder_state_dict': cross_encoder.state_dict(),
                'cross_encoder_optimizer_state_dict': cross_encoder_optimizer.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(directory_to_save, "ce_checkpoint.pth"))

            print(f"Best model checkpoint saved with validation loss: {best_val_loss} in directory {directory_to_save}")

            print(f"Best model saved with validation loss: {best_val_loss}")

            # del best_cross_encoder_state_dict
            # torch.cuda.empty_cache()
        # else:
        #     num_epochs_without_improvement += 1

        # if num_epochs_without_improvement >= patience:
        #     print(f"Early stopping triggered. No improvement in validation loss for {patience} consecutive epochs.")
        #     break
        if early_stopping(avg_val_loss):
            print("Early stopping triggered")
            break

    print("Training complete.")
    return best_model



# Calculate total training steps and set num_warmup_steps as a fraction of total steps
total_steps = (len(train_dataset) // (config["batch_size"] * config["gradient_accumulation_steps"])) * config["epochs"]
config["num_warmup_steps"] = int(0.1 * total_steps)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


best_model = train_dpr_model(config, train_dataset, validation_dataset, t5_cross_encoder, t5_tokenizer, device)

