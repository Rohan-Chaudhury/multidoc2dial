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

config = {
    "epochs": 40,
    "batch_size": 6,
    "learning_rates": {
        "question_encoder": 3e-5,
        "context_encoder": 3e-5,
        "cross_encoder": 1e-5
    },
    "gradient_accumulation_steps": 4096,
    "max_length": 512,
    "patience": 5,
    "temperature": 1.0,
}




class T5CrossEncoder(nn.Module):
    def __init__(self, pretrained_model_name, model_max_length, dropout_rate = 0.1):
        super().__init__()
        config = T5Config.from_pretrained(pretrained_model_name)
        config.model_max_length = model_max_length
        self.t5 = T5EncoderModel.from_pretrained(pretrained_model_name, config=config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.t5.config.d_model, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits




class CustomDPRQuestionEncoderWithDropout(nn.Module):
    def __init__(self, model_name, dropout_rate):
        super(CustomDPRQuestionEncoderWithDropout, self).__init__()
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
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
        self.model = DPRContextEncoder.from_pretrained(model_name)
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
question_encoder = CustomDPRQuestionEncoderWithDropout("sivasankalpp/dpr-multidoc2dial-structure-question-encoder", 0.1)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")

context_encoder = CustomDPRContextEncoder(model_name="sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder", dropout_rate=0.1)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")


# reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
# reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Set this to the index of the GPU you want to use
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_encoder = nn.DataParallel(question_encoder)
context_encoder = nn.DataParallel(context_encoder)
question_encoder.to(device)
context_encoder.to(device)
t5_pretrained_model_name = "t5-large"
model_max_length = 512
t5_cross_encoder = T5CrossEncoder(t5_pretrained_model_name, model_max_length)
# t5_cross_encoder = T5CrossEncoder(t5_pretrained_model_name)
t5_cross_encoder= nn.DataParallel(t5_cross_encoder)
t5_cross_encoder.to(device)
t5_tokenizer = T5Tokenizer.from_pretrained(t5_pretrained_model_name, model_max_length=model_max_length)

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
    return remove_extra_spaces(question)




import random

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

# training_data= preprocess_data(training_data)
# test_data= preprocess_data(test_data)
# validation_data= preprocess_data(validation_data)

# from transformers import MarianMTModel, MarianTokenizer

# model_name = f'Helsinki-NLP/opus-mt-en-fr'
# en_fr_tokenizer = MarianTokenizer.from_pretrained(model_name)
# en_fr_model = MarianMTModel.from_pretrained(model_name).to(device)

# model_name = f'Helsinki-NLP/opus-mt-fr-en'
# fr_en_tokenizer = MarianTokenizer.from_pretrained(model_name)
# fr_en_model = MarianMTModel.from_pretrained(model_name).to(device)

# def backtranslate(texts, en_fr_model,en_fr_tokenizer,fr_en_model,fr_en_tokenizer, batch_size=32):
#     # Load the translation model and tokenizer

#     results = []
#     dataset = Dataset.from_dict({"text": texts})
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#     with torch.no_grad():
#         # iters = tqdm(range(0, len(texts) - batch_size + 1, batch_size),desc="Backtranslating")
#         for batch in tqdm(dataloader, desc="Translating"): 
#             batch_texts = batch["text"]
#             n = len(batch_texts)

#             inputs = en_fr_tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
#             translated = en_fr_model.generate(**inputs)
#             target_texts = en_fr_tokenizer.batch_decode(translated, skip_special_tokens=True)


#             inputs = fr_en_tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
#             backtranslated = fr_en_model.generate(**inputs)
#             source_texts = fr_en_tokenizer.batch_decode(backtranslated, skip_special_tokens=True)

#             results.extend(source_texts[:n])

#     return results


# Assuming `training_data` is a list of dictionaries containing "question", "positive_ctxs", "negative_ctxs", and "hard_negative_ctxs"
preprocessed_data = preprocess_data(training_data)
train_dataset = Dataset.from_dict(preprocessed_data)
train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

preprocessed_validation_data = preprocess_data(validation_data)
validation_dataset  = Dataset.from_dict(preprocessed_validation_data)
validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True)


# Add the preprocessing function that tokenizes the context and question
def preprocess_function(examples):
    question_encodings = question_tokenizer(examples['question'], truncation=True, padding='max_length', max_length=256)
    context_encodings = context_tokenizer(examples['context'], truncation=True, padding='max_length', max_length=256)
    
    encodings = {
        'input_ids': question_encodings['input_ids'],
        'attention_mask': question_encodings['attention_mask'],
        'context_input_ids': context_encodings['input_ids'],
        'context_attention_mask': context_encodings['attention_mask'],
        'labels': examples['labels'],
    }
    
    return encodings


import torch

from transformers import DPRQuestionEncoder, DPRContextEncoder
from torch.nn.utils import clip_grad_norm_


def kaiming_initialization(module):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

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


def train_dpr_model(config, train_dataset, validation_dataset, model, cross_encoder, question_tokenizer, context_tokenizer, t5_tokenizer, device):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rates = config["learning_rates"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    num_warmup_steps = config["num_warmup_steps"]
    max_length = config["max_length"]
    patience = config["patience"]
    temperature = config["temperature"]

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

    optimizer_grouped_cross_encoder_parameters = [
        {
            "params": [p for n, p in cross_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in cross_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    early_stopping = EarlyStopping(patience=patience)
    cross_encoder_optimizer = AdamW(optimizer_grouped_cross_encoder_parameters, lr=learning_rates['cross_encoder'], eps=adam_epsilon)
    
    question_encoder_optimizer = AdamW(optimizer_grouped_question_encoder_parameters , lr=learning_rates['question_encoder'],  eps=adam_epsilon)
    context_encoder_optimizer = AdamW(optimizer_grouped_context_encoder_parameters, lr=learning_rates['context_encoder'],  eps=adam_epsilon)
    number_of_batches = ceil(len(train_dataset) / batch_size)
    total_steps = number_of_batches * epochs // gradient_accumulation_steps

    base_learning_rate = 1e-5

    T_0 = 2000  # Number of iterations for the first restart
    T_mult = 2  # Multiplicative factor to increase the cycle length after each restart

    question_encoder_scheduler = CosineAnnealingWarmRestarts(question_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)

    context_encoder_scheduler = CosineAnnealingWarmRestarts(context_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)

    cross_encoder_scheduler = CosineAnnealingWarmRestarts(cross_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)

    # model.to(device)
    best_val_loss = float('inf')
    best_model = None

    # Initialize loss weights
    contrastive_weight = 1 / 3
    cross_encoder_weight = 1 / 3
    dr_weight = 1 / 3

    # Initialize running averages for losses
    running_contrastive_loss = 0.0
    running_cross_encoder_loss = 0.0
    running_dr_loss = 0.0

    # Define the smoothing factor for updating the running averages
    smoothing_factor = 0.1

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0

        alpha = 0.2
        train_iter=tqdm(train_dataloader, desc="Training", ncols=100)
        for idx, batch in enumerate(train_iter):
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = process_batch(batch, question_tokenizer, context_tokenizer, max_length, device)
            # example_indices = shuffled_indices[i:i + batch_size]

            questions = batch["question"]
            positive_contexts = batch["positive_context"]
            negative_contexts = batch["negative_context"]

            # Compute embeddings
            anchor_embeddings = model.question_encoder(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = model.context_encoder(positive_input_ids, positive_attention_mask)
            negative_embeddings = model.context_encoder(negative_input_ids, negative_attention_mask)

            # Compute contrastive loss
            loss = contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            positive_scores = compute_scores(anchor_embeddings, positive_embeddings)
            negative_scores = compute_scores(anchor_embeddings, negative_embeddings)
            margin = 1.0
            dr_loss = torch.clamp(margin - positive_scores + negative_scores, min=0).mean()
            # Prepare cross-encoder inputs
            positive_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, positive_contexts)]
            negative_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, negative_contexts)]
            t5_input = positive_t5_input + negative_t5_input

            t5_encodings = t5_tokenizer(t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
            t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)

            cross_encoder_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()
            cross_encoder_labels = torch.tensor([1] * len(questions) + [0] * len(questions), dtype=torch.float, device=device)
            cross_encoder_loss = F.binary_cross_entropy_with_logits(cross_encoder_logits, cross_encoder_labels)


            # Update running averages for losses
            running_contrastive_loss = (1 - smoothing_factor) * running_contrastive_loss + smoothing_factor * loss.item()
            running_cross_encoder_loss = (1 - smoothing_factor) * running_cross_encoder_loss + smoothing_factor * cross_encoder_loss.item()
            running_dr_loss = (1 - smoothing_factor) * running_dr_loss + smoothing_factor * dr_loss.item()

            # Update weights based on the running averages of losses
            total_running_loss = running_contrastive_loss + running_cross_encoder_loss + running_dr_loss
            contrastive_weight = running_contrastive_loss / total_running_loss
            cross_encoder_weight = running_cross_encoder_loss / total_running_loss
            dr_weight = running_dr_loss / total_running_loss

            # Compute the combined loss using updated weights
            combined_loss = contrastive_weight * loss + cross_encoder_weight * cross_encoder_loss + dr_weight * dr_loss
            total_loss += combined_loss.item()



        # Combine the losses
            # combined_loss = (1 - alpha) * loss + alpha * cross_encoder_loss + dr_loss
            # total_loss += combined_loss.item()

            # total_loss += loss.item()

            combined_loss = combined_loss / gradient_accumulation_steps
            combined_loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                # Apply gradient clipping
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                clip_grad_norm_(cross_encoder.parameters(), max_norm=1.0)

                question_encoder_optimizer.step()
                context_encoder_optimizer.step()
                cross_encoder_optimizer.step()

                question_encoder_scheduler.step()
                context_encoder_scheduler.step()
                cross_encoder_scheduler.step()

                model.zero_grad()
                cross_encoder.zero_grad()

            
            train_iter.set_description(f"Training (loss = {loss.item():.4f})")
            train_iter.refresh()

        avg_train_loss = total_loss / number_of_batches
        print(f"Training loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0.0
        number_of_batches_validation = ceil(len(validation_dataset) / batch_size)

        val_iter=tqdm(validation_dataloader, desc="Validation", ncols=100)
        for batch in val_iter:
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = process_batch(batch, question_tokenizer, context_tokenizer, max_length, device)
            questions = batch["question"]
            positive_contexts = batch["positive_context"]
            negative_contexts = batch["negative_context"]
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

            # Prepare cross-encoder inputs
            positive_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, positive_contexts)]
            negative_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, negative_contexts)]
            t5_input = positive_t5_input + negative_t5_input

            t5_encodings = t5_tokenizer(t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
            t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)

            with torch.no_grad():
                cross_encoder_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()
                cross_encoder_labels = torch.tensor([1] * len(questions) + [0] * len(questions), dtype=torch.float, device=device)
                cross_encoder_loss = F.binary_cross_entropy_with_logits(cross_encoder_logits, cross_encoder_labels)


            # Combine the losses
            combined_loss = contrastive_weight * loss + cross_encoder_weight * cross_encoder_loss + dr_weight * dr_loss


            total_val_loss += combined_loss.item()
            val_iter.set_description(f"Validation (loss = {loss.item():.4f})")
            val_iter.refresh()
            
        avg_val_loss = total_val_loss / number_of_batches_validation
        print(f"Validation loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_cross_encoder_state_dict = copy.deepcopy(cross_encoder.state_dict())
            # Get the current date and time as a string
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            directory_to_save="/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/output/contrastive_discriminative_pre_trained/"+timestamp
            # Create a new directory with the timestamp
            os.makedirs(directory_to_save, exist_ok=True)

            # Save the model checkpoint with additional metadata
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state_dict,
                'cross_encoder_state_dict': best_cross_encoder_state_dict,
                'question_encoder_optimizer_state_dict': question_encoder_optimizer.state_dict(),
                'context_encoder_optimizer_state_dict': context_encoder_optimizer.state_dict(),
                'cross_encoder_optimizer_state_dict': cross_encoder_optimizer.state_dict(),
                'question_encoder_scheduler_state_dict': question_encoder_scheduler.state_dict(),
                'context_encoder_scheduler_state_dict': context_encoder_scheduler.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(directory_to_save, "checkpoint.pth"))

            print(f"Best model checkpoint saved with validation loss: {best_val_loss} in directory {directory_to_save}")

            print(f"Best model saved with validation loss: {best_val_loss}")
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


combined_model = DPRCombinedModel(question_encoder, context_encoder)
combined_model.apply(kaiming_initialization)
best_model = train_dpr_model(config, train_dataset, validation_dataset, combined_model, t5_cross_encoder,question_tokenizer, context_tokenizer, t5_tokenizer, device)


# import torch
# from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRConfig
# from dpr_model import DPRCombinedModel

# def load_saved_model(checkpoint_path, device):
#     question_encoder_config = DPRConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
#     context_encoder_config = DPRConfig.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

#     question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base", config=question_encoder_config)
#     context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base", config=context_encoder_config)

#     combined_model = DPRCombinedModel(question_encoder, context_encoder)

#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     combined_model.load_state_dict(checkpoint['model_state_dict'])
#     combined_model.to(device)
#     combined_model.eval()

#     return combined_model

# # Load the saved model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# checkpoint_path = "your_saved_checkpoint_directory/checkpoint.pth"
# loaded_model = load_saved_model(checkpoint_path, device)


# import torch
# from transformers import BertModel, T5ForConditionalGeneration

# # Initialize models
# question_encoder = BertModel.from_pretrained("bert-base-uncased")
# context_encoder = BertModel.from_pretrained("bert-base-uncased")
# cross_encoder = T5ForConditionalGeneration.from_pretrained("t5-base")

# # Load checkpoint
# checkpoint_path = "path/to/checkpoint.pth"
# checkpoint = torch.load(checkpoint_path)

# # Load state_dicts
# question_encoder.load_state_dict(checkpoint["question_encoder_state_dict"])
# context_encoder.load_state_dict(checkpoint["context_encoder_state_dict"])
# cross_encoder.load_state_dict(checkpoint["cross_encoder_state_dict"])

# # Set models to eval mode
# question_encoder.eval()
# context_encoder.eval()
# cross_encoder.eval()
