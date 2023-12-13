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
from transformers import DPRConfig, DPRContextEncoder, DPRContextEncoderTokenizer, T5Config, T5EncoderModel, T5Tokenizer, AutoTokenizer, T5Model
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

from tqdm.auto import tqdm

from torch.nn.functional import cosine_similarity
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # Set this to the index of the GPU you want to use
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import random
from transformers import T5EncoderModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import json
import re
from datasets import Dataset


config = {
    "epochs": 20,
    "batch_size": 1,
    "gradient_accumulation_steps": 256,
    "max_length": 512,
    "patience": 20,
    "temperature": 1.0,
}



with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/training_psg_data.json", "r") as f:
    training_data = json.load(f)

# with open("/home/softlab/Documents/md2d_trial/to_zip/dpr.psg.multidoc2dial_all.structure.json", "r") as f:
#     corpus_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/validation_psg_data.json", "r") as f:
    validation_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/test_psg_data.json", "r") as f:
    test_data = json.load(f)


from torch.nn import HingeEmbeddingLoss

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Model, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from torch.nn import functional as F
from transformers import get_cosine_schedule_with_warmup


class MyDataset(Dataset):
    def __init__(self, queries, pos_contexts, neg_contexts, question_tokenizer, context_tokenizer, max_length):
        self.queries = queries
        self.pos_contexts = pos_contexts
        self.neg_contexts = neg_contexts
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        pos_context = self.pos_contexts[idx]
        neg_contexts = self.neg_contexts[idx]

        query_encoding = self.question_tokenizer(query, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        pos_context_encoding = self.context_tokenizer(pos_context, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        neg_contexts_encoding = [self.context_tokenizer(neg_context, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt') for neg_context in neg_contexts]

        # Add attention masks
        query_attention_mask = query_encoding['attention_mask'].flatten()
        pos_context_attention_mask = pos_context_encoding['attention_mask'].flatten()
        neg_contexts_attention_mask = [neg_context_encoding['attention_mask'].flatten() for neg_context_encoding in neg_contexts_encoding]

        return {
            'query': query_encoding['input_ids'].flatten(),
            'query_attention_mask': query_attention_mask,
            'pos_context': pos_context_encoding['input_ids'].flatten(),
            'pos_context_attention_mask': pos_context_attention_mask,
            'neg_contexts': [neg_context_encoding['input_ids'].flatten() for neg_context_encoding in neg_contexts_encoding],
            'neg_contexts_attention_mask': neg_contexts_attention_mask
        }

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


class CustomDPRContextEncoder(nn.Module):
    def __init__(self, model_name, dropout_rate=0.1):
        super(CustomDPRContextEncoder, self).__init__()
        self.model = DPRContextEncoder.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return pooled_output




class CustomDPRQuestionEncoderWithDropout(nn.Module):
    def __init__(self, model_name, dropout_rate=0.1):
        super(CustomDPRQuestionEncoderWithDropout, self).__init__()
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return pooled_output



class DPRCombinedModel(nn.Module):
    def __init__(self, question_encoder: CustomDPRQuestionEncoderWithDropout, context_encoder: CustomDPRContextEncoder):
        super(DPRCombinedModel, self).__init__()
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder

    def forward(self, question_input_ids, question_attention_mask, context_input_ids, context_attention_mask):
        question_outputs = self.question_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask)
        context_outputs = self.context_encoder(input_ids=context_input_ids, attention_mask=context_attention_mask)
        return question_outputs, context_outputs


# checkpoint_path_dpr = "/home/softlab/Documents/md2d_trial/scp_trial/loss_19.5_best/dpr_checkpoint.pth"

# config = DPRConfig.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
question_encoder = CustomDPRQuestionEncoderWithDropout("sivasankalpp/dpr-multidoc2dial-structure-question-encoder", 0.1)
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")

context_encoder = CustomDPRContextEncoder(model_name="sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder", dropout_rate=0.1)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")


checkpoint_path_dpr = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/best_model_state.bin"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
question_encoder = nn.DataParallel(question_encoder)
context_encoder = nn.DataParallel(context_encoder)

question_encoder.to(device)
question_encoder.train()
context_encoder.to(device)
context_encoder.train()
model = DPRCombinedModel(question_encoder, context_encoder)
model.to(device)
    # checkpoint_dpr = torch.load(checkpoint_path_dpr, map_location=device)
    # print ("Loss: ", checkpoint_dpr['loss'])
    # combined_model.load_state_dict(checkpoint_dpr['model_state_dict'], strict=False)
    # combined_model.to(device)
    # combined_model.eval()

    # return combined_model

checkpoint_dpr = torch.load(checkpoint_path_dpr, map_location=device)
# print ("Loss: ", checkpoint_dpr['loss'])
# model.load_state_dict(checkpoint_dpr, strict=False)
model.load_state_dict(checkpoint_dpr)
print ("Using strict=False")
# combined_model = load_saved_model(checkpoint_path_dpr)
print ("Model Loaded")

del checkpoint_dpr
print ("Checkpoint Deleted")


    

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, margin=2):
        super(CombinedLoss, self).__init__()
        self.margin = margin


    def forward(self, query_vectors, positive_vectors, negative_vectors):
        dot_product_loss = self.compute_dot_product_loss(query_vectors, positive_vectors, negative_vectors)
        return dot_product_loss

    def compute_dot_product_loss(self, query_vectors, positive_vectors, negative_vectors):
        positive_similarities = torch.matmul(query_vectors, positive_vectors.t())
        negative_similarities = [torch.matmul(query_vectors, neg_vectors.t()) for neg_vectors in negative_vectors]
        losses = [F.relu(self.margin - positive_similarities + negative_similarity) for negative_similarity in negative_similarities]
        loss = sum(losses) / len(losses)
        return loss.mean() if len(loss.size()) > 0 else loss


from torch_optimizer import Lookahead
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm.auto import tqdm

def validate(model, dataloader, loss_fn, device):
    model.eval()
    model.context_encoder.eval()
    model.question_encoder.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", ncols=100):
            query_input_ids = batch['query'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            pos_context_input_ids = batch['pos_context'].to(device)
            pos_context_attention_mask = batch['pos_context_attention_mask'].to(device)
            neg_contexts_input_ids = [neg_context.to(device) for neg_context in batch['neg_contexts']]
            neg_contexts_attention_masks = [neg_attention_mask.to(device) for neg_attention_mask in batch['neg_contexts_attention_mask']]

            query_outputs, pos_context_outputs = model(query_input_ids, query_attention_mask, pos_context_input_ids, pos_context_attention_mask)
            # neg_contexts_outputs = [model.context_model(neg_context_input_ids, neg_context_attention_mask).last_hidden_state[:, 0, :] for neg_context_input_ids, neg_context_attention_mask in zip(neg_contexts_input_ids, neg_contexts_attention_mask)]
            
            neg_contexts_outputs = [model.context_encoder(neg_context_input_id, neg_context_attention_mask) for neg_context_input_id, neg_context_attention_mask in zip(neg_contexts_input_ids, neg_contexts_attention_masks)]

            loss = loss_fn(query_outputs, pos_context_outputs, neg_contexts_outputs)

            total_loss += loss.item()

    return total_loss / len(dataloader)

def train(model, train_dataloader, val_dataloader, loss_fn, optimizer, device, scheduler, num_epochs, accumulation_steps):
    model.train()

    best_loss = float('inf')
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        epoch_loss = 0.0
        train_iterator = tqdm(train_dataloader, desc="Training", ncols=100)
        for i, batch in enumerate(train_iterator):
            query_input_ids = batch['query'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            pos_context_input_ids = batch['pos_context'].to(device)
            pos_context_attention_mask = batch['pos_context_attention_mask'].to(device)
            neg_contexts_input_ids = [neg_context.to(device) for neg_context in batch['neg_contexts']]
            neg_contexts_attention_masks = [neg_attention_mask.to(device) for neg_attention_mask in batch['neg_contexts_attention_mask']]

            query_outputs, pos_context_outputs = model(query_input_ids, query_attention_mask, pos_context_input_ids, pos_context_attention_mask)
            # neg_contexts_outputs = [model.context_model(neg_context_input_ids, neg_context_attention_mask).last_hidden_state[:, 0, :] for neg_context_input_ids, neg_context_attention_mask in zip(neg_contexts_input_ids, neg_contexts_attention_mask)]
            neg_contexts_outputs = [model.context_encoder(neg_context_input_id, neg_context_attention_mask) for neg_context_input_id, neg_context_attention_mask in zip(neg_contexts_input_ids, neg_contexts_attention_masks)]


            loss = loss_fn(query_outputs, pos_context_outputs, neg_contexts_outputs)
            loss_to_print = loss.item()
            loss = loss / accumulation_steps
            loss.backward()

            if (i+1) % accumulation_steps == 0  or (i + 1 == len(train_dataloader)):  
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_iterator.set_description(f"Training (loss = {loss_to_print:.4f})")
            train_iterator.refresh()

            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        print(f'Training loss: {epoch_loss}')

        val_loss = validate(model, val_dataloader, loss_fn, device)
        print(f'Validation loss: {val_loss}')


        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model_state.bin')
            print("Best model saved with validation loss:", best_loss)


def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_question(question):
    return remove_extra_spaces(question)



def preprocess_data(training_data, negative_weight=30):

    queries = []
    pos_contexts = []
    neg_contexts = []

    for item in training_data:
        # print ("\nQuestion: \n"+ item["question"])
        question = preprocess_question(item["question"])
        # print ("\nQuestion changed: \n"+ question)
        positive_ctxs = item["positive_psg"]
        negative_ctxs = item["negative_psgs"][:negative_weight]

        # print ("\nPositive Context: \n"+ positive_ctxs[0]["text"])
        positive_context = remove_extra_spaces(positive_ctxs["text"])
        # print ("\nPositive Context changed: \n"+ positive_context)

        # print ("\nNegative Context: \n"+ negative_ctxs[0]["text"])
        # print ("\nNegative Context changed: \n"+ remove_extra_spaces(negative_ctxs[0]))

        # all_negative_ctxs = (negative_ctxs * negative_weight) + (hard_negative_ctxs * hard_negative_weight)

        negative_contexts = [remove_extra_spaces(negative_ctx) for negative_ctx in negative_ctxs]
        # print ("\nNegative Context changed: \n"+ str(negative_contexts))
        # for negative_context in negative_contexts:
        #     print ("\nNegative Context changed: \n"+ negative_context)
        queries.append(question)
        pos_contexts.append(positive_context)
        neg_contexts.append(negative_contexts)
               

    return queries, pos_contexts, neg_contexts

# Usage
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NEGATIVE_WEIGHT = 26

queries, pos_contexts, neg_contexts = preprocess_data(training_data, negative_weight=NEGATIVE_WEIGHT)


dataset = MyDataset(queries, pos_contexts, neg_contexts, question_tokenizer, context_tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

loss_fn = CombinedLoss().to(device)


accumulation_steps = config["gradient_accumulation_steps"]


# ...

base_optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

optimizer = Lookahead(base_optimizer)


epochs = config['epochs']
steps_per_epoch = len(dataloader) // accumulation_steps


# Scheduler Initialization
num_training_steps = epochs * steps_per_epoch
num_warmup_steps = int(num_training_steps * 0.1) # 10% of train steps for warm-up
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)



queries_val, pos_contexts_val, neg_contexts_val = preprocess_data(validation_data, negative_weight=NEGATIVE_WEIGHT)
val_dataset = MyDataset(queries_val, pos_contexts_val, neg_contexts_val, question_tokenizer, context_tokenizer, max_length=512)
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
print ("Validation data loaded with equal weights and not normalized, running loss is used for dynamic weights fixed")
print ("Training Started with negative weight of: ", NEGATIVE_WEIGHT)
print ("batch size: ", config["batch_size"])
print ("epochs: ", config["epochs"])
print ("accumulation_steps: ", accumulation_steps)
print ("training with hard negative and some changes")
train(model, dataloader, val_dataloader, loss_fn, optimizer, device, scheduler, num_epochs=config['epochs'], accumulation_steps=accumulation_steps)



