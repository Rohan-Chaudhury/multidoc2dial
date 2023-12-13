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

from transformers import T5Config

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set this to the index of the GPU you want to use
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import random


import json
import re
from datasets import Dataset


t5_pretrained_model_name = "t5-large"
model_max_length = 512
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


def truncate_question_sequences(question, max_question_len=500):
    # print (question)
    question_tokens = t5_tokenizer(question, truncation=True, max_length=max_question_len, return_tensors="pt")
    # print(len(question_tokens["input_ids"].squeeze()))
    decoded_text = t5_tokenizer.decode(question_tokens["input_ids"].squeeze(), skip_special_tokens=True)
    print (decoded_text)
    # print ("\n")
    return decoded_text

def preprocess_question(question):
    return truncate_question_sequences(remove_extra_spaces(question), 500)




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


if os.path.exists("t5_training_data_flan.json"):
    with open("t5_training_data_flan.json", "r") as f:
        preprocessed_data = json.load(f)
else:
    preprocessed_data = preprocess_data(training_data)
    with open("t5_training_data_flan.json", "w") as f:
        json.dump(preprocessed_data, f)



if os.path.exists("t5_validation_data_flan.json"):
    with open("t5_validation_data_flan.json", "r") as f:
        preprocessed_validation_data = json.load(f)
else:
    preprocessed_validation_data = preprocess_data(validation_data, negative_weight=1, hard_negative_weight=2)
    with open("t5_validation_data_flan.json", "w") as f:
        json.dump(preprocessed_validation_data, f)


print ("\n\n Datasets loaded \n\n")


