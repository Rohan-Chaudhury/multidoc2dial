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
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Set this to the index of the GPU you want to use
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


with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.test.json", "r") as f:
    test_data = json.load(f)


def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def truncate_question_sequences(question, max_question_len=500):
    # print (question)
    question_tokens = t5_tokenizer(question, truncation=True, max_length=max_question_len, return_tensors="pt")
    # print(len(question_tokens["input_ids"].squeeze()))
    decoded_text = t5_tokenizer.decode(question_tokens["input_ids"].squeeze(), skip_special_tokens=True)
    # print (decoded_text)
    # print ("\n")
    return decoded_text

def preprocess_question(question):
    return truncate_question_sequences(remove_extra_spaces(question), 500)



def preprocess_data(training_data):
    questions=[]

    for item in tqdm(training_data):
        # print(item["question"])
        question = preprocess_question(item["question"])
        questions.append(question)
                

    return questions

t5_questions = preprocess_data(test_data)
print ("Length of questions: ", len(t5_questions))
import pickle

def save_list_to_file(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

save_list_to_file(t5_questions, 't5_questions_500.pkl')
