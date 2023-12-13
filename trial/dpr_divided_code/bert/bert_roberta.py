import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import re
import torch.nn as nn
import json
from tqdm import tqdm
import warnings
from torch.nn.utils import clip_grad_norm_
import logging
# logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
import torch_optimizer as optim
# warnings.filterwarnings("ignore")
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from math import ceil

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = {
    "epochs": 20,
    "batch_size": 1,
    "learning_rate": 1e-5, 
    "gradient_accumulation_steps": 4,
    "max_length": 512,
    "max_length_question": 128,
    "patience": 10,
    "temperature": 1.0,
    "negative_length": 7,
    "model_name": 'deepset/roberta-large-squad2',
    "weight_decay": 0.01,
    "model_path": "ce_model_roberta.pt"
}


def truncate_question(question, tokenizer, max_length_question):
    # # Tokenize the question
    # tokens_question = tokenizer.tokenize(question)

    # # Truncate tokens to the desired max_length
    # tokens_question = tokens_question[:max_length_question]
    # # print (len(tokens_question))

    # # Join tokens back into a single string
    # question_truncated = tokenizer.convert_tokens_to_string(tokens_question)
    
    # return question_truncated
    question_tokens = tokenizer(question, truncation=True, max_length=max_length_question, return_tensors="pt")
    # print("\n"+str(len(question_tokens["input_ids"].squeeze())))
    decoded_text = tokenizer.decode(question_tokens["input_ids"].squeeze(), skip_special_tokens=True)
    # print (decoded_text)
    # print ("\n")
    
    return decoded_text

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_question(question):
    return remove_extra_spaces(question)

# Define the re-ranker model
class Reranker(torch.nn.Module):
    def __init__(self, model_name, dropout_rate=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = self._init_classifier(self.bert.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)

    def _init_classifier(self, d_model):
        classifier = nn.Linear(d_model, 1)
        nn.init.kaiming_normal_(classifier.weight, nonlinearity='relu')
        return classifier


    # def forward(self, input_ids, attention_mask):
    #     outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
    #     cls_output = outputs.last_hidden_state[:, 0]
    #     # distance = torch.matmul(cls_output, self.vector)
    #     # return distance
    #     logits = self.classifier(cls_output)
    #     return logits

    def forward(self, input_ids, attention_mask):
        # Reshape the inputs to be two-dimensional
        # batch_size, num_passages, seq_length = input_ids.size()
        # input_ids = input_ids.view(-1, seq_length)
        # attention_mask = attention_mask.view(-1, seq_length)
        
        # Perform the forward pass through the BERT model
        # print("input_ids: ", input_ids.size())
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Reshape the outputs back to be three-dimensional
        # cls_output = cls_output.view(batch_size, num_passages, -1)
        # cls_output = self.layer_norm(cls_output)
        cls_output = self.dropout(cls_output)
        cls_output = self.layer_norm(cls_output)
        logits = self.classifier(cls_output).squeeze(-1)
        
        return logits


# Define the contrastive loss function
class ContrastiveLoss(torch.nn.Module):
    def forward(self, distances, labels):
        pos_distances = distances[labels == 1]
        neg_distances = distances[labels == 0]
        numerator = torch.exp(pos_distances)
        denominator = numerator + torch.sum(torch.exp(neg_distances))
        loss = -torch.log(numerator / denominator)
        return loss

# Define the dataset
class DialogDataset(Dataset):
    def __init__(self, data, tokenizer, max_length_question=128, max_length_total=512, negative_length=7):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length_question = max_length_question
        self.max_length_total = max_length_total
        self.negative_length = negative_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = preprocess_question(item["question"])
        question = truncate_question(question, self.tokenizer, self.max_length_question)
        positive_psg = remove_extra_spaces(item["positive_psg"]["text"])
        negative_psgs = [remove_extra_spaces(psg) for psg in item["negative_psgs"][:self.negative_length]]
        passages = [positive_psg] + negative_psgs
        labels = [1] + [0]*len(negative_psgs)
        encodings = self.tokenizer([question]*len(passages), passages, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length_total)
        return {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': torch.tensor(labels)}

def validate(model, valid_data_loader, loss_function, device, best_valid_loss, model_path, epoch):
        # Evaluate on the validation set
        model.eval()
        val_iter=tqdm(valid_data_loader, desc="Validation", ncols=100)

        valid_loss =  0.0
        with torch.no_grad():
            for batch in val_iter:
                input_ids = batch['input_ids'].to(device).squeeze(0)
                attention_mask = batch['attention_mask'].to(device).squeeze(0)
                labels = batch['labels'].to(device).squeeze(0)

                distances = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_function(distances, labels)
                valid_loss += loss.item()

                val_iter.set_description(f"Validation (loss = {loss.item():.4f})")
                val_iter.refresh()
        
        valid_loss = valid_loss / len(valid_data_loader)
        print(f'Epoch: {epoch+1}, Validation Loss: {valid_loss:.4f}')

        # Save the model if the validation loss decreases
        if valid_loss < best_valid_loss:
            print(f'Validation loss decreased ({best_valid_loss:.4f} --> {valid_loss:.4f}).  Saving model ...')
            torch.save(model.state_dict(), model_path)
            best_valid_loss = valid_loss

        model.train()
        return best_valid_loss


# Define the training loop
def train(model, data_loader, valid_data_loader, loss_function, optimizer, num_epochs, device, accumulation_steps, model_path, scheduler=None):
    best_valid_loss = float('inf')
    model.train()
    model.to(device)
    number_of_batches = ceil(len(train_dataset) / config["batch_size"])
    for epoch in range(num_epochs):
        model.train()
        print ("Model in training mode now 5")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0

        train_iter=tqdm(data_loader, desc="Training", ncols=100)
        for i, batch in enumerate(train_iter):
            input_ids = batch['input_ids'].to(device).squeeze(0)
            attention_mask = batch['attention_mask'].to(device).squeeze(0)
            labels = batch['labels'].to(device).squeeze(0)
            
            distances = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_function(distances, labels)
            # loss.backward()

            combined_loss =  loss 
            total_loss += combined_loss.item()

            combined_loss = combined_loss / accumulation_steps
            combined_loss.backward()
            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0  or (i + 1 == len(data_loader)):
                # clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            train_iter.set_description(f"Training (loss = {loss.item():.4f})")
            train_iter.refresh()
            
        avg_train_loss = total_loss / number_of_batches
        print(f"Training loss: {avg_train_loss}")

        best_valid_loss = validate(model, valid_data_loader, loss_function, device, best_valid_loss, model_path, epoch)



with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/training_psg_data.json", "r") as f:
    training_data = json.load(f)

# with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.psg.multidoc2dial_all.structure.json", "r") as f:
#     corpus_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/validation_psg_data.json", "r") as f:
    validation_data = json.load(f)

# with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.test.json", "r") as f:
#     test_data = json.load(f)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_name = config["model_name"]
print ("Loading the model: ", model_name)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocess the data
max_length_question = config["max_length_question"]
max_length_total = config["max_length"]
negative_length = config["negative_length"]
train_dataset = DialogDataset(training_data, tokenizer, max_length_question, max_length_total, negative_length)
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

validation_dataset = DialogDataset(validation_data, tokenizer, max_length_question, max_length_total, negative_length)
validation_loader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True)

# Initialize the model, loss function, and optimizer
model = Reranker(model_name=model_name)
loss_function = ContrastiveLoss()
# optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
adam_epsilon = 1e-8 
# no_decay = ["bias", "LayerNorm.weight"]

# optimizer_grouped_cross_encoder_parameters = [
#     {
#         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": config["weight_decay"],
#     },
#     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
# ]

optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],eps=adam_epsilon, weight_decay=config["weight_decay"])
# optimizer = optim.Lookahead(optimizer)

eta_min = 1e-6  # Minimum learning rate during cosine annealing
T_0 = 2000  # Number of iterations for the first restart
T_mult = 2  # Multiplicative factor to increase the cycle length after each restart

# Initialize the learning rate scheduler
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

# Train the model
num_epochs = config["epochs"]
train(model, train_loader, validation_loader, loss_function, optimizer, num_epochs, device, accumulation_steps=config["gradient_accumulation_steps"], model_path=config["model_path"], scheduler=scheduler)
