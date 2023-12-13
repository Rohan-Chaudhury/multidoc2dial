
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoModel, AutoTokenizer
import re
import json


class SingleModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)[1]
        return self.linear(outputs).squeeze(-1)

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_question(question):
    return remove_extra_spaces(question)

def truncate_question(question, tokenizer, max_length_question):
    # Tokenize the question
    tokens_question = tokenizer.tokenize(question)

    # Truncate tokens to the desired max_length
    tokens_question = tokens_question[:max_length_question]

    # Join tokens back into a single string
    question_truncated = tokenizer.convert_tokens_to_string(tokens_question)

    return question_truncated

class ReRankerDataset(Dataset):
    def __init__(self, data, tokenizer, max_length_question, max_length_total):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length_question = max_length_question
        self.max_length_total = max_length_total

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = preprocess_question(item["question"])
        question = truncate_question(question, self.tokenizer, self.max_length_question)
        positive_psg = remove_extra_spaces(item["positive_psg"]["text"])
        negative_psgs = [remove_extra_spaces(psg) for psg in item["negative_psgs"]]
        passages = [positive_psg] + negative_psgs
        labels = [1] + [0]*len(negative_psgs)
        encodings = self.tokenizer([question]*len(passages), passages, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length_total)
        return {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': torch.tensor(labels)}





model_name = 'deepset/roberta-large-squad2'  # Choose the model you prefer
model = SingleModel(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/training_psg_data.json", "r") as f:
    training_data = json.load(f)

# with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.psg.multidoc2dial_all.structure.json", "r") as f:
#     corpus_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/validation_psg_data.json", "r") as f:
    validation_data = json.load(f)

max_length_question = 128
max_length_total = 512
train_dataset = ReRankerDataset(training_data, tokenizer, max_length_question, max_length_total)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

validation_dataset = ReRankerDataset(validation_data, tokenizer, max_length_question, max_length_total)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(10):  # Number of epochs
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
