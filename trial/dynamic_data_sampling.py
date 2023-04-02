import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

def get_uncertainty(model, dataset, device):
    model.eval()
    uncertainty_scores = []
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=-1)
            uncertainty = 1 - probabilities.max(dim=-1).values
            uncertainty_scores.extend(uncertainty.tolist())
    
    return uncertainty_scores

def train_dynamic(model, dataset, device, epochs, uncertainty_threshold):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        uncertainty_scores = get_uncertainty(model, dataset, device)
        uncertain_samples = [i for i, score in enumerate(uncertainty_scores) if score >= uncertainty_threshold]
        train_indices, val_indices = train_test_split(uncertain_samples, test_size=0.1)
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, batch)
            loss.backward()
            optimizer.step()
        
        # Evaluate the model on the validation subset
        # ...

Dynamic data sampling - Uncertainty sampling:
For uncertainty sampling, you will need to calculate the model's uncertainty on each example in your dataset, and then sample the most uncertain examples for training. Here's a possible implementation: 


In this example, train_dynamic trains the model using only the most uncertain examples, based on the uncertainty threshold provided. You will need to adapt the `compute_loss` function to your use case. You can also use a different sampling strategy, such as entropy sampling, or a combination of both.
In this example, train_dynamic trains the model using only the most uncertain examples, based on the uncertainty threshold provided. You will need to adapt the `compute_loss


from sklearn.metrics import accuracy_score

def compute_loss(logits, batch):
    # Define your loss computation based on your specific task
    # ...
    return loss

def evaluate(model, val_subset, device):
    model.eval()
    dataloader = DataLoader(val_subset, batch_size=16, shuffle=False)
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            pred_labels = logits.argmax(dim=-1).tolist()
            true_labels.extend(batch['labels'].tolist())
            predictions.extend(pred_labels)

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

def train_dynamic(model, dataset, device, epochs, uncertainty_threshold):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        uncertainty_scores = get_uncertainty(model, dataset, device)
        uncertain_samples = [i for i, score in enumerate(uncertainty_scores) if score >= uncertainty_threshold]
        train_indices, val_indices = train_test_split(uncertain_samples, test_size=0.1)
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            logits = model(input_ids, attention_mask)
            loss = compute_loss(logits, batch)
            loss.backward()
            optimizer.step()
        
        val_accuracy = evaluate(model, val_subset, device)
        print(f"Epoch: {epoch + 1}, Validation accuracy: {val_accuracy:.4f}")
