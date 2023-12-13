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


def truncate_question_sequences(question):
    # print (question)
    question_tokens = t5_tokenizer(question, return_tensors="pt")
    # print(len(question_tokens["input_ids"].squeeze()))

    # print ("\n")
    return (len(question_tokens["input_ids"].squeeze()))

def preprocess_question(question):
    return truncate_question_sequences(remove_extra_spaces(question))




def preprocess_data(training_data, negative_weight=1, hard_negative_weight=1):
    lengths=[]

    for item in tqdm(training_data):
        # print(item["question"])
        question = preprocess_question(item["question"])
        lengths.append(question)

    return lengths



preprocessed_data = preprocess_data(validation_data)
print (max(preprocessed_data))
print (min(preprocessed_data))
print (sum(preprocessed_data)/len(preprocessed_data))



