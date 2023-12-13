class EnsembleModel(torch.nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.models = torch.nn.ModuleList([AutoModel.from_pretrained(name) for name in model_names])
        self.tokenizers = [AutoTokenizer.from_pretrained(name) for name in model_names]
        self.linear = torch.nn.Linear(len(model_names), 1)

    def forward(self, input_ids, attention_mask):
        outputs = [model(input_ids=input_ids, attention_mask=attention_mask)[1] for model in self.models]
        outputs = torch.stack(outputs, dim=-1)
        return self.linear(outputs).squeeze(-1)

class ReRankerDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        positive_psg = item["positive_psg"]["text"]
        negative_psgs = [psg["text"] for psg in item["negative_psgs"]]
        passages = [positive_psg] + negative_psgs
        labels = [1] + [0]*len(negative_psgs)
        encodings = self.tokenizer([question]*len(passages), passages, return_tensors='pt', padding=True, truncation=True, max_length=128)
        return {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask'], 'labels': torch.tensor(labels)}

model_names = ['bert-base-uncased', 'roberta-base', 'google/electra-base-discriminator']
model = EnsembleModel(model_names)
tokenizer = AutoTokenizer.from_pretrained(model_names[0])  # Assuming all tokenizers are similar

# Assuming you have some data in this format
data = [...]  # Replace with your actual data

train_dataset = ReRankerDataset(data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optimizer = AdamW(list(model.parameters()) + list(model.linear.parameters()), lr=1e-5)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(10):  # Number of epochs
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        optimizer.step()
