import os
import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer
from sklearn.metrics import f1_score, recall_score, mean_reciprocal_rank

def load_checkpoint(checkpoint_path, model, cross_encoder):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    cross_encoder.load_state_dict(checkpoint["cross_encoder_state_dict"])
    print(f"Loaded model checkpoint with validation loss: {checkpoint['loss']}")

def evaluate_model(test_dataset, model, cross_encoder, question_tokenizer, context_tokenizer, t5_tokenizer, device):
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model.eval()
    cross_encoder.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = process_batch(batch, question_tokenizer, context_tokenizer, max_length, device)
            
            # Compute question and context embeddings
            question_embeddings = model.question_encoder(anchor_input_ids, anchor_attention_mask)
            context_embeddings = model.context_encoder(positive_input_ids, positive_attention_mask)

            # Compute scores and predictions
            scores = compute_scores(question_embeddings, context_embeddings)
            predictions = torch.topk(scores, k=10, dim=1).indices

            # Compute cross-encoder scores
            t5_input = [f"{q} </s> {c}" for q, c in zip(batch["question"], batch["positive_context"])]
            t5_encodings = t5_tokenizer(t5_input, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)
            cross_encoder_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()

            # Calculate evaluation metrics
            preds.append(predictions.cpu().numpy())
            labels.append(batch["label"].cpu().numpy())

    f1 = f1_score(labels, preds, average='weighted')
    recall_1 = recall_score(labels, preds, average=None, k=1)
    recall_5 = recall_score(labels, preds, average=None, k=5)
    recall_10 = recall_score(labels, preds, average=None, k=10)
    mrr = mean_reciprocal_rank(labels, preds)

    print(f"F1 score: {f1}")
    print(f"Recall @1: {recall_1}")
    print(f"Recall @5: {recall_5}")
    print(f"Recall @10: {recall_10}")
    print(f"MRR: {mrr}")

# Load the checkpoint and initialize the models
checkpoint_path = "path/to/checkpoint.pth"
model = CustomDPRQuestionEncoderWithDropout(config)
cross_encoder = T5CrossEncoder(config)
load_checkpoint(checkpoint_path, model, cross_encoder)

# Evaluate the models on the test dataset
test_dataset =
