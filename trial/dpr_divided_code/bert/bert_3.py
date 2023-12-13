import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import re
import torch.nn as nn
import json
from tqdm import tqdm
import warnings
import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

warnings.filterwarnings("ignore")

from math import ceil

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = {
    "epochs": 10,
    "batch_size": 2,
    "learning_rate": 1e-5, 
    "gradient_accumulation_steps": 2,
    "max_length": 512,
    "max_length_question": 128,
    "patience": 10,
    "temperature": 1.0,
    "negative_length": 7,
    "model_name": 'google/electra-base-discriminator',
    "weight_decay": 0.01,
    "model_path": "ce_model_roberta.pt"
}


def truncate_question(question, tokenizer, max_length_question):
    # Tokenize the question
    tokens_question = tokenizer.tokenize(question)

    # Truncate tokens to the desired max_length
    tokens_question = tokens_question[:max_length_question]
    # print (len(tokens_question))

    # Join tokens back into a single string
    question_truncated = tokenizer.convert_tokens_to_string(tokens_question)
    
    return question_truncated


def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_question(question):
    return remove_extra_spaces(question)

# Define the re-ranker model
class Reranker(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        self.classifier = self._init_classifier(self.bert.config.hidden_size)

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
        batch_size, num_passages, seq_length = input_ids.size()
        input_ids = input_ids.view(-1, seq_length)
        attention_mask = attention_mask.view(-1, seq_length)
        
        # Perform the forward pass through the BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        
        # Reshape the outputs back to be three-dimensional
        cls_output = cls_output.view(batch_size, num_passages, -1)
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

# Define the training loop
def train(model, data_loader, valid_data_loader, loss_function, optimizer, num_epochs, device, accumulation_steps, model_path):
    best_valid_loss = float('inf')
    model.train()
    model.to(device)
    number_of_batches = ceil(len(train_dataset) / config["batch_size"])
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0

        train_iter=tqdm(data_loader, desc="Training", ncols=100)
        for i, batch in enumerate(train_iter):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            distances = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_function(distances, labels)
            # loss.backward()

            combined_loss =  loss 
            total_loss += combined_loss.item()

            combined_loss = combined_loss / accumulation_steps
            combined_loss.backward()
            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0  or (i + 1 == len(data_loader)):
                optimizer.step()
                optimizer.zero_grad()
            train_iter.set_description(f"Training (loss = {loss.item():.4f})")
            train_iter.refresh()
            
        avg_train_loss = total_loss / number_of_batches
        print(f"Training loss: {avg_train_loss}")
        # Evaluate on the validation set
        model.eval()

        total_val_loss = 0.0
        number_of_batches_validation = ceil(len(validation_dataset) / config["batch_size"])

        val_iter=tqdm(valid_data_loader, desc="Validation", ncols=100)

        valid_loss =  0.0
        with torch.no_grad():
            for batch in val_iter:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

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


with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/training_psg_data.json", "r") as f:
    training_data = json.load(f)

# with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.psg.multidoc2dial_all.structure.json", "r") as f:
#     corpus_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/validation_psg_data.json", "r") as f:
    validation_data = json.load(f)


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
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

# Train the model
num_epochs = config["epochs"]
train(model, train_loader, validation_loader, loss_function, optimizer, num_epochs, device, accumulation_steps=config["gradient_accumulation_steps"], model_path=config["model_path"])
