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

import json
from torch.optim.lr_scheduler import CyclicLR
from transformers import DPRConfig, DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRReader, DPRReaderTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from tqdm.auto import tqdm

from transformers import AutoModel

# Fast Gradient Sign Method (FGSM) as an adversarial attack:
def decode(self, logits):
    input_ids = torch.argmax(logits, dim=-1)
    return input_ids

def fgsm_attack(input_ids, attention_mask, labels, model, loss_fn, epsilon):
    input_ids = input_ids.clone().detach().to(device)
    attention_mask = attention_mask.clone().detach().to(device)

    input_ids.requires_grad = False
    attention_mask.requires_grad = False

    logits, _ = model.question_encoder(input_ids=input_ids, attention_mask=attention_mask)

    # Create a new tensor that shares storage with logits and set requires_grad to True
    logits_perturb = logits.detach().clone()
    logits_perturb.requires_grad = True
    
    loss = loss_fn(logits_perturb, labels)
    model.zero_grad()
    loss.backward()

    gradients = logits_perturb.grad.data
    perturbation = gradients.sign() * epsilon

    # Apply the perturbation to the logits
    adversarial_logits = logits + perturbation

    # Run the model in evaluation mode to get the adversarial predictions
    model.eval()
    with torch.no_grad():
        adversarial_logits = adversarial_logits.unsqueeze(0)
        adversarial_input_ids = input_ids + epsilon * perturbation.sign()

    model.train()

    return adversarial_input_ids, attention_mask




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
    def decode(self, logits):
        input_ids = torch.argmax(logits, dim=-1)
        return input_ids


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
    def decode(self, logits):
        input_ids = torch.argmax(logits, dim=-1)
        return input_ids

# config = DPRConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = CustomDPRQuestionEncoderWithDropout("facebook/dpr-question_encoder-single-nq-base", 0.1)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

context_encoder = CustomDPRContextEncoder(model_name="facebook/dpr-ctx_encoder-single-nq-base", dropout_rate=0.1)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

reader = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
reader_tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # Set this to the index of the GPU you want to use
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

# corpus_data_dict = preprocess_corpus_data(corpus_data)

corpus_data_dict = preprocess_corpus_data(corpus_data)
corpus_dataset = Dataset.from_dict(corpus_data_dict)

# training_data= preprocess_data(training_data)
# test_data= preprocess_data(test_data)
# validation_data= preprocess_data(validation_data)

from transformers import MarianMTModel, MarianTokenizer

model_name = f'Helsinki-NLP/opus-mt-en-fr'
en_fr_tokenizer = MarianTokenizer.from_pretrained(model_name)
en_fr_model = MarianMTModel.from_pretrained(model_name).to(device)

model_name = f'Helsinki-NLP/opus-mt-fr-en'
fr_en_tokenizer = MarianTokenizer.from_pretrained(model_name)
fr_en_model = MarianMTModel.from_pretrained(model_name).to(device)

def backtranslate(texts, en_fr_model,en_fr_tokenizer,fr_en_model,fr_en_tokenizer, batch_size=32):
    # Load the translation model and tokenizer

    results = []
    dataset = Dataset.from_dict({"text": texts})
    dataloader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        # iters = tqdm(range(0, len(texts) - batch_size + 1, batch_size),desc="Backtranslating")
        for batch in tqdm(dataloader, desc="Translating"): 
            batch_texts = batch["text"]
            n = len(batch_texts)

            inputs = en_fr_tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            translated = en_fr_model.generate(**inputs)
            target_texts = en_fr_tokenizer.batch_decode(translated, skip_special_tokens=True)


            inputs = fr_en_tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            backtranslated = fr_en_model.generate(**inputs)
            source_texts = fr_en_tokenizer.batch_decode(backtranslated, skip_special_tokens=True)

            results.extend(source_texts[:n])

    return results


# training_data= preprocess_data(training_data)
# test_data= preprocess_data(test_data)
# validation_data= preprocess_data(validation_data)

training_data= preprocess_data(training_data[:10])
test_data= preprocess_data(test_data[:10])
validation_data= preprocess_data(validation_data[:10])

# do_it=True

# if do_it or not os.path.exists("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/output/combined_training_data.json"):
#     training_data= preprocess_data(training_data)
#     # Apply backtranslation to your training data
#     backtranslated_questions = backtranslate(training_data["question"], en_fr_model,en_fr_tokenizer,fr_en_model,fr_en_tokenizer) 
#     backtranslated_contexts = backtranslate(training_data["context"], en_fr_model,en_fr_tokenizer,fr_en_model,fr_en_tokenizer) 
#     backtranslated_labels = training_data["labels"]

#     training_data["question"].extend(backtranslated_questions)
#     training_data["context"].extend(backtranslated_contexts)
#     training_data["labels"].extend(backtranslated_labels)


#     with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/output/combined_training_data.json", "w") as outfile:
#         json.dump(training_data, outfile, ensure_ascii=False, indent=2)



# with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/output/combined_training_data.json", "r") as infile:
#     loaded_data = json.load(infile)



# train_dataset = Dataset.from_dict(loaded_data)

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

def train_dpr_model(config, train_dataset, validation_dataset, model, question_tokenizer, context_tokenizer, device, adverserial_attack=True):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rates = config["learning_rates"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    num_warmup_steps = config["num_warmup_steps"]
    max_length = config["max_length"]
    patience = config["patience"]
    epsilon = config["epsilon"]


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
    # max_learning_rate = 1e-4
    # step_size_up = 2000

    # question_encoder_scheduler = CyclicLR(question_encoder_optimizer, base_lr=base_learning_rate, max_lr=max_learning_rate, step_size_up=step_size_up, mode='triangular')


    # context_encoder_scheduler = CyclicLR(context_encoder_optimizer, base_lr=base_learning_rate, max_lr=max_learning_rate, step_size_up=step_size_up, mode='triangular')

    # question_encoder_scheduler = get_linear_schedule_with_warmup(question_encoder_optimizer, num_warmup_steps, total_steps)
    # context_encoder_scheduler = get_linear_schedule_with_warmup(context_encoder_optimizer, num_warmup_steps, total_steps)

    T_0 = 2000  # Number of iterations for the first restart
    T_mult = 2  # Multiplicative factor to increase the cycle length after each restart

    question_encoder_scheduler = CosineAnnealingWarmRestarts(question_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)

    context_encoder_scheduler = CosineAnnealingWarmRestarts(context_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)
    model.to(device)
    best_val_loss = float('inf')
    best_model = None

    # num_epochs_without_improvement = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0

        # shuffled_indices = list(range(len(train_dataset)))
        # random.shuffle(shuffled_indices)

        # train_iter = tqdm(range(0, len(train_dataset) - batch_size + 1, batch_size), desc="Training", ncols=100)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_iter=tqdm(train_loader, desc="Training", ncols=100)
        for idx, batch in enumerate(train_iter):

            # example_indices = shuffled_indices[i:i + batch_size]
            questions = batch["question"]
            contexts = batch["context"]
            labels = torch.tensor(batch["labels"], dtype=torch.long).to(device)

            # Pass the entire batch of questions and contexts to the tokenizers
            question_encodings = question_tokenizer(questions, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            context_encodings = context_tokenizer(contexts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)


            input_ids = question_encodings['input_ids'].to(device)
            attention_mask = question_encodings['attention_mask'].to(device)
            context_input_ids = context_encodings['input_ids'].to(device)
            context_attention_mask = context_encodings['attention_mask'].to(device)

            if adverserial_attack:
                # Apply the FGSM attack to the input_ids and attention_mask
                adversarial_input_ids, adversarial_attention_mask = fgsm_attack(input_ids, attention_mask, labels, model, F.cross_entropy, epsilon)
                # Unsqueeze the tensors to add an extra dimension
                adversarial_input_ids = adversarial_input_ids.unsqueeze(0)
                adversarial_attention_mask = adversarial_attention_mask.unsqueeze(0)

                # Modify the attention_mask tensor to match the attention_scores tensor's size
                extended_attention_mask = adversarial_attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

                
                # Calculate the adversarial loss
                question_outputs_adv, context_outputs_adv = model(question_input_ids=adversarial_input_ids, question_attention_mask=extended_attention_mask, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask)

                scores_adv = torch.matmul(question_outputs_adv, context_outputs_adv.T)
                loss_adv = F.cross_entropy(scores_adv, labels)


            # Calculate the normal loss
            question_outputs_normal, context_outputs_normal = model(question_input_ids=input_ids, question_attention_mask=attention_mask, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask)

            scores_normal = torch.matmul(question_outputs_normal, context_outputs_normal.T)
            loss_normal = F.cross_entropy(scores_normal, labels)
            loss = loss_normal + loss_adv
            total_loss += loss.item()

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
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
        # val_iter = tqdm(range(0, len(validation_dataset) - batch_size + 1, batch_size), desc="Validation", ncols=100)

        val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        val_iter=tqdm(val_dataloader, desc="Validation", ncols=100)
        for batch in val_iter:
            # Collect questions, contexts, and labels from the current batch
            questions = batch["question"]
            contexts = batch["context"]
            labels = torch.tensor(batch["labels"], dtype=torch.long).to(device)

            # Pass the entire batch of questions and contexts to the tokenizers
            question_encodings = question_tokenizer(questions, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            context_encodings = context_tokenizer(contexts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)


            input_ids = question_encodings['input_ids'].to(device)
            attention_mask = question_encodings['attention_mask'].to(device)
            context_input_ids = context_encodings['input_ids'].to(device)
            context_attention_mask = context_encodings['attention_mask'].to(device)

            with torch.no_grad():
                question_outputs, context_outputs = model(input_ids, attention_mask, context_input_ids, context_attention_mask)
                scores = torch.matmul(question_outputs, context_outputs.T)
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



config = {
    "epochs": 10,
    "batch_size": 32,
    "learning_rates": {
        "question_encoder": 3e-5,
        "context_encoder": 3e-5
    },
    "gradient_accumulation_steps": 32,
    "max_length": 512,
    "patience": 3,
    "epsilon": .05
}

# Calculate total training steps and set num_warmup_steps as a fraction of total steps
total_steps = (len(train_dataset) // (config["batch_size"] * config["gradient_accumulation_steps"])) * config["epochs"]
config["num_warmup_steps"] = int(0.1 * total_steps)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


combined_model = DPRCombinedModel(question_encoder, context_encoder)
combined_model.apply(kaiming_initialization)
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