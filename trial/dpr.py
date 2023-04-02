import json
import torch
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast,
    DPRReader, DPRReaderTokenizerFast, TrainingArguments, Trainer
)

import transformers
print(transformers.__version__)

from transformers import DPRConfig, DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRReader, DPRReaderTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

config = DPRConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set this to the index of the GPU you want to use
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
    turns = question.split("[SEP]")
    questions=turns[0]
    turns=[turns[1]]
    turns = [turn.strip() for turn in turns if turn.strip()]
    turns = [turn.split("||") for turn in turns]
    turns = [turn[::-1] for turn in turns]  # Reverse the order of previous turns
    turns = [" || ".join(turn) for turn in turns]

    return "Query: "+ questions.lower()+ " || Context: "+  " ".join(turns).lower() 




def preprocess_data(training_data):
    train_data = {
        "question": [],
        "context": [],
        "labels": []
    }
    
    for item in training_data:
        question = preprocess_question(item["question"])
        positive_ctxs = item["positive_ctxs"]
        negative_ctxs = item["negative_ctxs"]
        hard_negative_ctxs = item["hard_negative_ctxs"]
        
        for positive_ctx in positive_ctxs:
            context = remove_extra_spaces(positive_ctx["title"].lower() + " " + positive_ctx["text"].lower())
            train_data["question"].append(question)
            train_data["context"].append(context)
            train_data["labels"].append(2)
        
        for negative_ctx in negative_ctxs:
            context = remove_extra_spaces(negative_ctx["title"].lower() + " " + negative_ctx["text"].lower())
            train_data["question"].append(question)
            train_data["context"].append(context)
            train_data["labels"].append(1)
        
        for hard_negative_ctx in hard_negative_ctxs:
            context = remove_extra_spaces(hard_negative_ctx["title"].lower() + " " + hard_negative_ctx["text"].lower())
            train_data["question"].append(question)
            train_data["context"].append(context)
            train_data["labels"].append(0)
    
    return train_data

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

corpus_data_dict = preprocess_corpus_data(corpus_data)
corpus_dataset = Dataset.from_dict(corpus_data_dict)

training_data= preprocess_data(training_data)
test_data= preprocess_data(test_data)
validation_data= preprocess_data(validation_data)

train_dataset = Dataset.from_dict(training_data)
test_dataset = Dataset.from_dict(test_data)
validation_dataset = Dataset.from_dict(validation_data)

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
import torch.nn as nn
from transformers import DPRQuestionEncoder, DPRContextEncoder
from torch.nn.utils import clip_grad_norm_
class DPRCombinedModel(nn.Module):
    def __init__(self, question_encoder: DPRQuestionEncoder, context_encoder: DPRContextEncoder):
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

def train_dpr_model(config, train_dataset, validation_dataset, model, question_tokenizer, context_tokenizer, device):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rates = config["learning_rates"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    num_warmup_steps = config["num_warmup_steps"]
    max_length = config["max_length"]
    patience = config["patience"]


    question_encoder_optimizer = AdamW(model.question_encoder.parameters(), lr=learning_rates['question_encoder'])
    context_encoder_optimizer = AdamW(model.context_encoder.parameters(), lr=learning_rates['context_encoder'])
    number_of_batches = ceil(len(train_dataset) / batch_size)
    total_steps = number_of_batches * epochs // gradient_accumulation_steps
    question_encoder_scheduler = get_linear_schedule_with_warmup(question_encoder_optimizer, num_warmup_steps, total_steps)
    context_encoder_scheduler = get_linear_schedule_with_warmup(context_encoder_optimizer, num_warmup_steps, total_steps)

    model.to(device)
    best_val_loss = float('inf')
    best_model = None

    num_epochs_without_improvement = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0

        shuffled_indices = list(range(len(train_dataset)))
        random.shuffle(shuffled_indices)

        train_iter = tqdm(range(0, len(train_dataset) - batch_size + 1, batch_size), desc="Training", ncols=100)

        for i in train_iter:

            example_indices = shuffled_indices[i:i + batch_size]

            question_encodings = question_tokenizer([train_dataset[example_idx]["question"] for example_idx in example_indices], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            context_encodings = context_tokenizer([train_dataset[example_idx]["context"] for example_idx in example_indices], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            labels = torch.tensor([train_dataset[example_idx]["labels"] for example_idx in example_indices], dtype=torch.long).to(device)

            input_ids = question_encodings['input_ids'].to(device)
            attention_mask = question_encodings['attention_mask'].to(device)
            context_input_ids = context_encodings['input_ids'].to(device)
            context_attention_mask = context_encodings['attention_mask'].to(device)
            question_outputs, context_outputs = model(input_ids, attention_mask, context_input_ids, context_attention_mask)

            scores = torch.matmul(question_outputs.pooler_output, context_outputs.pooler_output.T)

            loss = F.cross_entropy(scores, labels)

            total_loss += loss.item()

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (i + 1) % gradient_accumulation_steps == 0:
                # Apply gradient clipping
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                question_encoder_optimizer.step()
                context_encoder_optimizer.step()
                question_encoder_scheduler.step()
                context_encoder_scheduler.step()

                model.zero_grad()
            
            train_iter.set_description(f"Training (loss = {loss.item():.4f})")
            train_iter.refresh()

        avg_train_loss = total_loss / number_of_batches
        print(f"Training loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0.0
        number_of_batches_validation = ceil(len(validation_dataset) / batch_size)
        val_iter = tqdm(range(0, len(validation_dataset) - batch_size + 1, batch_size), desc="Validation", ncols=100)
        for i in val_iter:

            question_encodings = question_tokenizer([validation_dataset[example_idx]["question"] for example_idx in range(i, i + batch_size)], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            context_encodings = context_tokenizer([validation_dataset[example_idx]["context"] for example_idx in range(i, i + batch_size)], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            labels = torch.tensor([validation_dataset[example_idx]["labels"] for example_idx in range(i, i + batch_size)], dtype=torch.long).to(device)

            input_ids = question_encodings['input_ids'].to(device)
            attention_mask = question_encodings['attention_mask'].to(device)
            context_input_ids = context_encodings['input_ids'].to(device)
            context_attention_mask = context_encodings['attention_mask'].to(device)

            with torch.no_grad():
                question_outputs, context_outputs = model(input_ids, attention_mask, context_input_ids, context_attention_mask)
                scores = torch.matmul(question_outputs.pooler_output, context_outputs.pooler_output.T)
                loss = F.cross_entropy(scores, labels)

                total_val_loss += loss.item()
            val_iter.set_description(f"Validation (loss = {loss.item():.4f})")
            val_iter.refresh()
            
        avg_val_loss = total_val_loss / number_of_batches_validation
        print(f"Validation loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())

            # Get the current date and time as a string
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            directory_to_save="/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/output/"+timestamp
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
            }, os.path.join(directory_to_save, "checkpoint.pth"))

            print(f"Best model checkpoint saved with validation loss: {best_val_loss} in directory {directory_to_save}")

            print(f"Best model saved with validation loss: {best_val_loss}")
        else:
            num_epochs_without_improvement += 1

        if num_epochs_without_improvement >= patience:
            print(f"Early stopping triggered. No improvement in validation loss for {patience} consecutive epochs.")
            break

    print("Training complete.")
    return best_model


config = {
    "epochs": 10,
    "batch_size": 16,
    "learning_rates": {
        "question_encoder": 5e-5,
        "context_encoder": 5e-5
    },
    "gradient_accumulation_steps": 16,
    "max_length": 512,
    "patience": 3
}

# Calculate total training steps and set num_warmup_steps as a fraction of total steps
total_steps = (len(train_dataset) // (config["batch_size"] * config["gradient_accumulation_steps"])) * config["epochs"]
config["num_warmup_steps"] = int(0.1 * total_steps)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


combined_model = DPRCombinedModel(question_encoder, context_encoder)
best_model = train_dpr_model(config, train_dataset, validation_dataset, combined_model, question_tokenizer, context_tokenizer, device)


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