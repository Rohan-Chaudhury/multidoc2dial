import os
GPUS="3"

os.environ["WANDB_DISABLED"] = "true"

os.environ["CUDA_VISIBLE_DEVICES"] = GPUS


import pandas as pd
import datasets
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import EarlyStoppingCallback
from tqdm import tqdm
import os
import json
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch_optimizer as optim
import re
from datasets import Dataset
import random
random.seed(42)

config = LongformerConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 15
# LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 1024
EVAL_BATCH_SIZE = BATCH_SIZE
EFF_BATCH_SIZE = 32
NUM_GPUS = len(GPUS.split(','))
print ("Number of GPUs are: ", NUM_GPUS)
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
#linear warmup steps equal to 0.1 of the total training steps

print ("Gradient accumulation steps are: ", EFF_BATCH_SIZE//(BATCH_SIZE*NUM_GPUS))
# load model and tokenizer and define length of the text sequence
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                           gradient_checkpointing=False,
                                                           attention_window = 256,
                                                           attention_probs_dropout_prob = 0.2,
                                                              hidden_dropout_prob = 0.2)
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = 1024)



with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/training_psg_data.json", "r") as f:
    training_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/validation_psg_data.json", "r") as f:
    validation_data = json.load(f)



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
    question = truncate_question_sequences(question, tokenizer, max_question_len=256)
    question = question.replace("[SEP]", " [HISTORY] ")
    question = remove_extra_spaces(question)
    return question


def preprocess_data(training_data, negative_weight=30):
    train_data = {
        "input_text": [],  # concatenated text of question + context
        "label": []  # 1 for positive, 0 for negative
    }

    for item in tqdm(training_data):
        question = preprocess_question(item["question"])
        positive_ctx = remove_extra_spaces(item["positive_psg"]["text"])
        
        # Positive samples
        train_data["input_text"].append(f"[QUESTION] {question} </s> [CONTEXT] {positive_ctx}")
        train_data["label"].append(1)  # Positive context

        negative_ctxs = item["negative_psgs"][:negative_weight]

        for negative_ctx in negative_ctxs:
            negative_context = remove_extra_spaces(negative_ctx)

            # Negative samples
            train_data["input_text"].append(f"[QUESTION] {question} </s> [CONTEXT] {negative_context}")
            train_data["label"].append(0)  # Negative context
 
    combined = list(zip(train_data["input_text"], train_data["label"]))
    random.shuffle(combined)
    train_data["input_text"], train_data["label"] = zip(*combined)

    return train_data

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

print ("Doing data processing now for longformer truncated--! \n")
train_data = preprocess_data(training_data, negative_weight=10)

validation_data = preprocess_data(validation_data, negative_weight=30)
print ("\n new datasets loaded ----\n")

from torch.nn.functional import one_hot
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, tokenizer, inputs, outputs, max_length=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.inputs = inputs
        self.outputs = outputs
        self.max_length = max_length

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_label = torch.tensor(self.outputs[idx])
        # print ("Output label is: ", output_label)
        input_tokenized = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        global_attention_mask = create_global_attention_mask(input_text, tokenizer)
        # print ("Length of global attention mask is: ", len(torch.tensor(global_attention_mask).squeeze()))
        # print ("Length of input tokenized is: ", len(input_tokenized["input_ids"].squeeze()))
        # labels = labels["input_ids"].masked_fill(labels["input_ids"] == self.tokenizer.pad_token_id, -100)
        return {
            "input_ids": input_tokenized["input_ids"].squeeze(),
            "attention_mask": input_tokenized["attention_mask"].squeeze(),
            "labels": output_label,
            "global_attention_mask": torch.tensor(global_attention_mask).squeeze()
        }

    def __len__(self):
        return len(self.inputs)

train_dataset = TranslationDataset(tokenizer, train_data['input_text'], train_data['label'])
val_dataset = TranslationDataset(tokenizer, validation_data['input_text'], validation_data['label'])





# define the training arguments
training_args = TrainingArguments(
    output_dir='./results_weighted_loss_function',
    num_train_epochs = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = EFF_BATCH_SIZE//(BATCH_SIZE*NUM_GPUS),    
    per_device_eval_batch_size= EVAL_BATCH_SIZE,
    evaluation_strategy = "epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    save_total_limit = 5,
    save_strategy="epoch",
    weight_decay=WEIGHT_DECAY,
    logging_steps = 4,
    learning_rate=LEARNING_RATE,
    seed=42,
    logging_dir='./logs',
    dataloader_num_workers = 0,
    run_name = 'first_try_weighted_loss',
    remove_unused_columns=False,
)


from transformers import Trainer

import torch.nn as nn


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0], device=device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)


print ("New NEw Training the model now weighted new new no lookahead---- \n")
# Train the model
trainer.train()


trainer.evaluate()