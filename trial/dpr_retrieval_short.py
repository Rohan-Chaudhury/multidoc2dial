
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DPRQuestionEncoder, DPRContextEncoder
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import rankdata
from tqdm import tqdm
import numpy as np

import json
import torch
from transformers import (
    DPRContextEncoder, DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast,
    DPRReader, DPRReaderTokenizerFast, TrainingArguments, Trainer
)
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import transformers
from torch.optim import SGD
print(transformers.__version__)
from transformers import T5Tokenizer
import json
from torch.optim.lr_scheduler import CyclicLR
from transformers import DPRConfig, DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRReader, DPRReaderTokenizer
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from tqdm.auto import tqdm
from transformers import T5EncoderModel
from transformers import AutoModel

import re
def preprocess_question(question):
    # turns = question.split("[SEP]")
    # questions=turns[0]
    # turns=[turns[1]]
    # turns = [turn.strip() for turn in turns if turn.strip()]
    # turns = [turn.split("||") for turn in turns]
    # turns = [turn[::-1] for turn in turns]  # Reverse the order of previous turns
    # turns = [" || ".join(turn) for turn in turns]

    # return "Query: "+ questions.lower()+ " || Context: "+  " ".join(turns).lower()
    return question

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_data(training_data, negative_weight=1, hard_negative_weight=2):
    train_data = {
        "question": [],
        "positive_context": [],
        "negative_context": []
    }

    for item in training_data:
        question = preprocess_question(item["question"])
        positive_ctxs = item["positive_ctxs"]
        negative_ctxs = item["negative_ctxs"]
        hard_negative_ctxs = item["hard_negative_ctxs"]

        for positive_ctx in positive_ctxs:
            positive_context = remove_extra_spaces(positive_ctx["title"].lower() + " " + positive_ctx["text"].lower())

        # Combine negative_ctxs and hard_negative_ctxs for sampling
        all_negative_ctxs = negative_ctxs + hard_negative_ctxs 

        negative_context= []
        for negative_ctx in all_negative_ctxs:
            negative_context.append(remove_extra_spaces(negative_ctx["title"].lower() + " " + negative_ctx["text"].lower()))

        train_data["question"].append(question)
        train_data["positive_context"].append(positive_context)
        train_data["negative_context"].append(negative_context)

    return train_data


# def process_batch(batch, question_tokenizer, context_tokenizer, max_length, device):
    # questions = batch["question"]
    # positive_contexts = batch["positive_context"]
    # negative_contexts = batch["negative_context"]

def process_batch(questions,positive_contexts, negative_contexts,  question_tokenizer, context_tokenizer,max_length, device):

    anchor_encodings = question_tokenizer(questions, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    positive_encodings = context_tokenizer(positive_contexts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    negative_encodings=[]
    
    # for b_negative_context in negative_contexts:
    #     batch_negative_contexts = []
    #     for negative_context in b_negative_context:
    #         batch_negative_contexts.append[context_tokenizer(negative_context, return_tensors='pt', padding=True, truncation=True, max_length=max_length)]
    #     negative_encodings.append(batch_negative_contexts)

    for negative_context in negative_contexts:
        negative_encodings.append(context_tokenizer(negative_context, return_tensors='pt', padding=True, truncation=True, max_length=max_length))

    anchor_input_ids, anchor_attention_mask = anchor_encodings['input_ids'].to(device), anchor_encodings['attention_mask'].to(device)
    positive_input_ids, positive_attention_mask = positive_encodings['input_ids'].to(device), positive_encodings['attention_mask'].to(device)
    # negative_input_ids =[]
    # negative_attention_mask = []
    # for negative_encoding in negative_encodings:
    #     negative_input_ids.append(negative_encoding['input_ids'].to(device))
    #     negative_attention_mask.append(negative_encoding['attention_mask'].to(device))

    return anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_encodings


class DPRCombinedModel(nn.Module):
    def __init__(self, question_encoder: DPRQuestionEncoder, context_encoder: DPRContextEncoder):
        super(DPRCombinedModel, self).__init__()
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder

    def forward(self, question_input_ids, question_attention_mask, context_input_ids, context_attention_mask):
        question_outputs = self.question_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask)
        context_outputs = self.context_encoder(input_ids=context_input_ids, attention_mask=context_attention_mask)
        return question_outputs, context_outputs

def evaluate(model, question_tokenizer, context_tokenizer, data_loader, device):
    # model.eval()
    model.eval()
    f1_scores = []
    mrr_scores = []
    r_at_1 = []
    r_at_5 = []
    r_at_10 = []
    em_scores = []
    print(len(data_loader["question"]))
    with torch.no_grad():
        # with tqdm(data_loader, desc="Evaluating") as pbar:
            for i in range(len(data_loader["question"])):
                # print(batch)
                # question_inputs = {k: v.to(device) for k, v in batch["question_inputs"].items()}
                # positive_ctx_inputs = {k: v.to(device) for k, v in batch["positive_ctx_inputs"].items()}
                # negative_ctx_inputs_list = batch["negative_ctx_inputs_list"]
                # question_outputs, positive_ctx_outputs = model(
                #     question_inputs["input_ids"], question_inputs["attention_mask"],
                #     positive_ctx_inputs["input_ids"], positive_ctx_inputs["attention_mask"]
                # )

                anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_encodings = process_batch(data_loader["question"][i], data_loader["positive_context"][i], data_loader["negative_context"][i],question_tokenizer, context_tokenizer, 512, device)
                # example_indices = shuffled_indices[i:i + batch_size]


                # Compute embeddings
                anchor_embeddings = model.question_encoder(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = model.context_encoder(positive_input_ids, positive_attention_mask)
                negative_embeddings = []
                # for b_negative_encoding in negative_encodings:
                #     batch_negative_embeddings = []
                #     for negative_encoding in b_negative_encoding:
                #         batch_negative_embeddings.append(model.context_encoder(negative_encoding["input_ids"].to(device), negative_encoding["attention_mask"].to(device)))
                #     negative_embeddings.append(batch_negative_embeddings)

                for negative_encoding in negative_encodings:
                    negative_embeddings.append(model.context_encoder(negative_encoding["input_ids"].to(device), negative_encoding["attention_mask"].to(device)))

                similarity_scores = [torch.matmul(anchor_embeddings.pooler_output, positive_embeddings.pooler_output.T).tolist()]

                for negative_embedding in negative_embeddings:
                    # for negative_embedding in b_negative_embedding:
                        similarity_scores.append(torch.matmul(anchor_embeddings.pooler_output, negative_embedding.pooler_output.T).tolist())

                ranks = rankdata(similarity_scores, method="max")
                positive_ctx_rank = ranks[0]
                # print(positive_ctx_rank)
                binary_true = [1] + [0] * len(negative_embedding)
                binary_pred = [1 if rank == positive_ctx_rank else 0 for rank in range(len(binary_true))]
                f1_scores.append(f1_score(binary_true, binary_pred))
                
                mrr_scores.append(1 / positive_ctx_rank)
                r_at_1.append(1 if positive_ctx_rank == 1 else 0)
                r_at_5.append(1 if positive_ctx_rank <= 5 else 0)
                r_at_10.append(1 if positive_ctx_rank <= 10 else 0)

                # Calculate EM (Exact Match) score
                em_scores.append(1 if positive_ctx_rank == 1 else 0)
                # pbar.set_postfix({"F1": np.mean(f1_scores), "MRR": np.mean(mrr_scores), "R@1": np.mean(r_at_1), "R@5": np.mean(r_at_5), "R@10": np.mean(r_at_10), "EM": np.mean(em_scores)})
                # print("done")
                if i%100 == 0:
                    print(i)


    f1_avg = np.mean(f1_scores)
    mrr_avg = np.mean(mrr_scores)
    r_at_1_avg = np.mean(r_at_1)
    r_at_5_avg = np.mean(r_at_5)
    r_at_10_avg = np.mean(r_at_10)
    em_avg = np.mean(em_scores)

    return {
        "f1": f1_avg,
        "mrr": mrr_avg,
        "r@1": r_at_1_avg,
        "r@5": r_at_5_avg,
        "r@10": r_at_10_avg,
        "em": em_avg
    }
# # Load model checkpoints
# question_encoder_checkpoint = torch.load("path/to/question_encoder_checkpoint.pth")
# context_encoder_checkpoint = torch.load("path/to/context_encoder_checkpoint.pth")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Set this to the index of the GPU you want to use
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the encoders with their respective checkpoints
# question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
# question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
# context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")


question_encoder = DPRQuestionEncoder.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
context_encoder = DPRContextEncoder.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder")

# sivasankalpp/dpr-multidoc2dial-structure-question-encoder
# sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder

# question_encoder.load_state_dict(question_encoder_checkpoint)
# context_encoder.load_state_dict(context_encoder_checkpoint)
question_encoder = nn.DataParallel(question_encoder)
context_encoder = nn.DataParallel(context_encoder)
question_encoder.to(device)
context_encoder.to(device)

# Create the DPR combined model
dpr_combined_model = DPRCombinedModel(question_encoder, context_encoder)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dpr_combined_model = dpr_combined_model.to(device)



# with open("/home/rohan/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.test.json", "r") as f:
#     test_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_dpr/dpr.multidoc2dial_all.structure.test.json", "r") as f:
    test_data = json.load(f)



test_data= preprocess_data(test_data[:200])
# print (test_data)
# test_dataset = Dataset.from_dict(test_data)
# test_data_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# Evaluate the model
evaluation_results = evaluate(dpr_combined_model, question_tokenizer, context_tokenizer, test_data, device)

print("Evaluation results:")
for metric, value in evaluation_results.items():
    print(f"{metric}: {value}")





# class DPRTestDataset(Dataset):
#     def __init__(self, data, question_tokenizer, context_tokenizer, max_length=512):
#         self.data = data
#         self.question_tokenizer = question_tokenizer
#         self.context_tokenizer = context_tokenizer
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data[idx]
#         question = preprocess_question(sample["question"])
#         positive_ctx = sample["positive_ctxs"][0]["text"]
#         negative_ctxs = [neg_ctx["text"] for neg_ctx in sample["negative_ctxs"]]
#         negative_ctxs.extend([hard_neg_ctx["text"] for hard_neg_ctx in sample["hard_negative_ctxs"]])
#         question_inputs = self.question_tokenizer(question, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
#         positive_ctx_inputs = self.context_tokenizer(positive_ctx, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
#         negative_ctx_inputs_list = [self.context_tokenizer(neg_ctx, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length) for neg_ctx in negative_ctxs]

#         return {
#             "question_inputs": {k: v.squeeze(0) for k, v in question_inputs.items()},
#             "positive_ctx_inputs": {k: v.squeeze(0) for k, v in positive_ctx_inputs.items()},
#             "negative_ctx_inputs_list": [{k: v.squeeze(0) for k, v in neg_ctx_inputs.items()} for neg_ctx_inputs in negative_ctx_inputs_list]
#         }
