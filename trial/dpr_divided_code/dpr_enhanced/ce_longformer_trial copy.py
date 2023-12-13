import pandas as pd
import datasets
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EarlyStoppingCallback
from tqdm import tqdm
import os
import json
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch_optimizer as optim
import re
from datasets import Dataset



os.environ["CUDA_VISIBLE_DEVICES"] = "1,0" 
config = LongformerConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 15
# LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 1024
EVAL_BATCH_SIZE = 16
EFF_BATCH_SIZE = 32
NUM_GPUS = 2
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
#linear warmup steps equal to 0.1 of the total training steps


# load model and tokenizer and define length of the text sequence
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096',
                                                           gradient_checkpointing=False,
                                                           attention_window = 256)
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
    truncated_text = tokenizer.decode(tokenized_question["input_ids"].squeeze(), skip_special_tokens=True)
    return truncated_text

def preprocess_question(question):
    question = truncate_question_sequences(question, tokenizer, max_question_len=256)
    question = remove_extra_spaces(question)
    question = question.replace("[SEP]", "[HISTORY]")
    return question


def preprocess_data(training_data, negative_weight=30):
    train_data = {
        "input_text": [],  # concatenated text of question + context
        "label": []  # 1 for positive, 0 for negative
    }

    for item in tqdm(training_data[:10]):
        question = preprocess_question(item["question"])
        positive_ctx = remove_extra_spaces(item["positive_psg"]["text"])
        
        # Positive samples
        train_data["input_text"].append(f"[QUESTION] {question} [CONTEXT] {positive_ctx}")
        train_data["label"].append(1)  # Positive context

        negative_ctxs = item["negative_psgs"][:negative_weight]

        for negative_ctx in negative_ctxs:
            negative_context = remove_extra_spaces(negative_ctx)

            # Negative samples
            train_data["input_text"].append(f"[QUESTION] {question} [CONTEXT] {negative_context}")
            train_data["label"].append(0)  # Negative context

    return train_data

def create_global_attention_mask(input_texts, tokenizer, max_len=MAX_SEQ_LEN):
    global_attention_masks = []
    for input_text in input_texts:
        tokens = tokenizer.tokenize(input_text)
        attention_mask = [0] * max_len  # initialize with zeros

        apply_attention = False
        for i, token in enumerate(tokens):
            if token in ["[QUESTION]", "[HISTORY]"]:
                apply_attention = True
            elif token == "[CONTEXT]":
                apply_attention = False

            if apply_attention:
                attention_mask[i] = 1

        global_attention_masks.append(attention_mask)
    
    return global_attention_masks


# def tokenization(batched_text):
#     tokenized_data = tokenizer(batched_text['input_text'], padding='longest', truncation=True, max_length=1024, return_tensors="pt")
#     tokenized_data['global_attention_mask'] = create_global_attention_mask(batched_text['input_text'], tokenizer)
#     return tokenized_data

print ("Doing data processing now for longformer truncated--! \n")
train_data = preprocess_data(training_data, negative_weight=14)

validation_data = preprocess_data(validation_data, negative_weight=30)
print ("\n new datasets loaded ----\n")

# train_data = Dataset.from_dict(train_data)
# train_data = train_data.map(tokenization, batched=True, batch_size=50)
# validation_data = Dataset.from_dict(validation_data)
# validation_data = validation_data.map(tokenization, batched = True, batch_size = 50)


# class CustomDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels

#     def __getitem__(self, idx):
#         item = {key: val[idx] for key, val in self.encodings.items()}
#         item["labels"] = torch.tensor([self.labels[idx]])
#         return item

#     def __len__(self):
#         return len(self.labels)


# train_dataset = CustomDataset(train_data, train_data['label'])
# val_dataset = CustomDataset(validation_data, validation_data['label'])
# def extract_encodings(hf_dataset):
#     return {k: hf_dataset[k] for k in hf_dataset.column_names if k not in ['label']}


# train_encodings = extract_encodings(train_data)
# val_encodings = extract_encodings(validation_data)

# train_dataset = CustomDataset(train_encodings, train_data['label'])
# val_dataset = CustomDataset(val_encodings, validation_data['label'])


# def tokenization(batched_text):
#     tokenized_data = tokenizer(batched_text['input_text'], padding='longest', truncation=True, max_length=1024, return_tensors="pt")
#     tokenized_data['global_attention_mask'] = create_global_attention_mask(batched_text['input_text'], tokenizer)
#     return tokenized_data

from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, tokenizer, inputs, outputs, max_length=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.inputs = inputs
        self.outputs = outputs
        self.max_length = max_length

    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        output_label = self.outputs[idx]

        input_tokenized = self.tokenizer(input_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        global_attention_mask = create_global_attention_mask(input_text, tokenizer)

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


# from typing import Dict, List
# from transformers.tokenization_utils_base import PreTrainedTokenizerBase
# from transformers.file_utils import PaddingStrategy
# import torch
# from transformers import DataCollatorWithPadding
# class CustomDataCollator(DataCollatorWithPadding):
#     def __init__(
#         self,
#         tokenizer: PreTrainedTokenizerBase,
#         padding: PaddingStrategy = True,
#         max_length: int = None,
#         pad_to_multiple_of: int = None,
#     ):
#         super().__init__(
#             tokenizer=tokenizer,
#             padding=padding,
#             max_length=max_length,
#             pad_to_multiple_of=pad_to_multiple_of,
#         )

#     def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
#         input_ids = [feature["input_ids"] for feature in features]

#         input_ids = self.tokenizer.pad(
#             {"input_ids": input_ids},
#             padding=self.padding,
#             max_length=self.max_length,
#             pad_to_multiple_of=self.pad_to_multiple_of,
#             return_tensors="pt",
#         )


#         input_ids = input_ids["input_ids"].masked_fill(input_ids["input_ids"] == self.tokenizer.pad_token_id, -100)

#         return {"input_ids": input_ids["input_ids"]}

# data_collator = CustomDataCollator(tokenizer)


# print ("length of input ids", len(train_data['input_ids'][0]))


# define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
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
    run_name = 'first_try',
    remove_unused_columns=False,
)




optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
optimizer = optim.Lookahead(optimizer)


num_training_steps = len(train_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size
num_warmup_steps = int(num_training_steps * 0.1)  

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=10,  
    early_stopping_threshold=0.0 
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    callbacks=[early_stopping_callback],
    optimizers=(optimizer, scheduler),
)

# Train the model
trainer.train()


trainer.evaluate()