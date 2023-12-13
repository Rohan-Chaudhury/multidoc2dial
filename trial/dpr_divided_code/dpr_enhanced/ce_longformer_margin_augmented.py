import os
GPUS="1,3"

os.environ["WANDB_DISABLED"] = "true"

os.environ["CUDA_VISIBLE_DEVICES"] = GPUS



from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments, LongformerConfig
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import EarlyStoppingCallback
from tqdm import tqdm
import os
import json
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch_optimizer as optim
import re
import random

import nltk
from nltk.corpus import wordnet
from random import choice
random.seed(42)

config = LongformerConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 15
# LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 1024
EVAL_BATCH_SIZE = 4
EFF_BATCH_SIZE = 64
NUM_GPUS = len(GPUS.split(','))
print ("Number of GPUs are: ", NUM_GPUS)
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
#linear warmup steps equal to 0.1 of the total training steps

print ("Gradient accumulation steps are: ", EFF_BATCH_SIZE//(BATCH_SIZE*NUM_GPUS))
# load model and tokenizer and define length of the text sequence
# config = LongformerConfig(num_labels=1)

config = LongformerConfig.from_pretrained('allenai/longformer-base-4096', num_labels=1)
config.gradient_checkpointing = False
config.attention_window = 256
config.attention_probs_dropout_prob = 0.2
config.hidden_dropout_prob = 0.2

model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', config=config)

model = model.to(device)

tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = 1024)

def replace_synonyms(text):
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    for i, (word, tag) in enumerate(tagged):
        # We'll replace only adjectives, adverbs, nouns and verbs for simplicity
        if tag.startswith(('JJ', 'RB', 'NN', 'VB')):
            synonyms = [syn.name().split('.')[0] for syn in wordnet.synsets(word) if syn.pos() == tag[0].lower()]
            if synonyms:
                tokens[i] = choice(synonyms)  # Randomly choose a synonym
    return ' '.join(tokens)



with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/training_psg_data.json", "r") as f:
    training_data = json.load(f)

with open("/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/dpr_divided_code/dpr_enhanced/to_hprc/enhanced_dpr/validation_psg_data.json", "r") as f:
    validation_data = json.load(f)



def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


def truncate_question_sequences(question, tokenizer, max_question_len=500):
    """
    Truncates a given question to a maximum length of tokens using the provided tokenizer.
    Returns the truncated text.
    """
    tokenized_question = tokenizer(question, truncation=True, max_length=max_question_len, return_tensors="pt")
    # print ("Length of tokenized question is: ", len(tokenized_question["input_ids"].squeeze()))
    truncated_text = tokenizer.decode(tokenized_question["input_ids"].squeeze(), skip_special_tokens=True)
    return truncated_text

def preprocess_question(question):
    question = truncate_question_sequences(question, tokenizer, max_question_len=256)
    question = question.replace("[SEP]", " [HISTORY] ")
    question = remove_extra_spaces(question)
    return question




# Modify the preprocess_data function to return pairs of positive and negative samples
def preprocess_data_for_pairs(training_data, negative_weight=30, augment_factor=30):
    positive_texts = []
    negative_texts = []

    for item in tqdm(training_data):
        question = preprocess_question(item["question"])
        positive_ctx = remove_extra_spaces(item["positive_psg"]["text"])

        if augment_factor > 0:
            positive_ctxs = []
            positive_ctxs.append(positive_ctx)
            # print ("\nPositive context is: ", positive_ctx)
            assert augment_factor == negative_weight-1
            for _ in range(augment_factor):
                try:
                    pos_title, pos_text = positive_ctx.split("//", 1)
                except:
                    print ("Positive context is: ", positive_ctx)

                augmented_text = replace_synonyms(pos_text)
                augmented_ctx =  remove_extra_spaces(f"{pos_title} // {augmented_text}")
                # print ("\n\nAugmented context is: ", augmented_ctx)
                # if pos_text.strip() != "" or augmented_text.strip() != "":
                #     while augmented_ctx in positive_ctxs:
                #         try:
                #             pos_title, pos_text = positive_ctx.split("//", 1)
                #         except:
                #             print ("Positive context is: ", positive_ctx)

                #         augmented_text = replace_synonyms(pos_text)
                #         augmented_ctx =  remove_extra_spaces(f"{pos_title} // {augmented_text}")
                # print ("\n\nAugmented context is: ", augmented_ctx)
                positive_ctxs.append(augmented_ctx)
            negative_ctxs = item["negative_psgs"][:negative_weight]
            assert len(positive_ctxs) == len(negative_ctxs)
            for positive_ctx_1, negative_ctx_1 in zip(positive_ctxs, negative_ctxs):
                negative_context = remove_extra_spaces(negative_ctx_1)
                positive_texts.append(f"[QUESTION] {question} </s> [CONTEXT] {positive_ctx_1}")
                negative_texts.append(f"[QUESTION] {question} </s> [CONTEXT] {negative_context}")

        else:
            negative_ctxs = item["negative_psgs"][:negative_weight]
            for negative_ctx in negative_ctxs:
                negative_context = remove_extra_spaces(negative_ctx)
                
                positive_texts.append(f"[QUESTION] {question} </s> [CONTEXT] {positive_ctx}")
                negative_texts.append(f"[QUESTION] {question} </s> [CONTEXT] {negative_context}")

    combined = list(zip(positive_texts, negative_texts))
    random.shuffle(combined)
    positive_texts, negative_texts = zip(*combined)

    return positive_texts, negative_texts

print ("Doing data processing now for longformer truncated new new--! \n")
NEGATIVE_WEIGHT = 10
positive_train, negative_train = preprocess_data_for_pairs(training_data, negative_weight=NEGATIVE_WEIGHT, augment_factor=NEGATIVE_WEIGHT-1)
positive_val, negative_val = preprocess_data_for_pairs(validation_data, negative_weight=30, augment_factor=0)

def create_global_attention_mask(input_text, tokenizer, max_len=MAX_SEQ_LEN):

    tokens = tokenizer.tokenize(input_text)
    attention_mask = [0] * max_len  # initialize with zeros

    apply_attention = True
    for i, token in enumerate(tokens):

        if token == "</s>":
            apply_attention = False
            # print ("token is: ", i)

        if apply_attention:
            attention_mask[i] = 1

    # print ("attention mask is: ", sum (attention_mask))
    return attention_mask


class PairwiseRankingDataset(Dataset):
    def __init__(self, tokenizer, positive_inputs, negative_inputs, max_length=MAX_SEQ_LEN):
        self.tokenizer = tokenizer
        self.positive_inputs = positive_inputs
        self.negative_inputs = negative_inputs
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def __getitem__(self, idx):
        # print ("idx is: ", idx)
        positive_text = self.positive_inputs[idx]
        negative_text = self.negative_inputs[idx]

        positive_tokenized = self.tokenizer(positive_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        positive_global_attention_mask = torch.tensor(create_global_attention_mask(positive_text, self.tokenizer))
        negative_tokenized = self.tokenizer(negative_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        negative_global_attention_mask = torch.tensor(create_global_attention_mask(negative_text, self.tokenizer))

        return {
            "input_ids": positive_tokenized["input_ids"].squeeze().to(self.device),
            "attention_mask": positive_tokenized["attention_mask"].squeeze().to(self.device),
            "global_attention_mask": positive_global_attention_mask.to(self.device),
            "negative_input_ids": negative_tokenized["input_ids"].squeeze().to(self.device),
            "negative_attention_mask": negative_tokenized["attention_mask"].squeeze().to(self.device),
            "negative_global_attention_mask": negative_global_attention_mask.to(self.device),
            "target": torch.tensor(1).to(self.device),  # Indicates positive should be ranked higher than negative
        }
    def __len__(self):
        return len(self.positive_inputs)



train_dataset = PairwiseRankingDataset(tokenizer, positive_train, negative_train)
val_dataset = PairwiseRankingDataset(tokenizer, positive_val, negative_val)








# define the training arguments
training_args = TrainingArguments(
    output_dir='./results_margin_loss_function',
    num_train_epochs = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = EFF_BATCH_SIZE//(BATCH_SIZE*NUM_GPUS),    
    per_device_eval_batch_size= EVAL_BATCH_SIZE,
    evaluation_strategy = "epoch",
    disable_tqdm = False, 
    load_best_model_at_end=True,
    save_total_limit = 5,
    save_strategy="epoch",
    weight_decay=WEIGHT_DECAY,
    logging_steps = 4,
    learning_rate=LEARNING_RATE,
    seed=42,
    logging_dir='./logs',
    dataloader_num_workers = 0,
    run_name = 'first_try_margin',
    remove_unused_columns=False,
)


criterion = nn.MarginRankingLoss(margin=1.0, reduction='mean').to(device)


from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput



from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, batch):
        model.train()
        positive_outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            global_attention_mask=batch["global_attention_mask"]
        ).logits.squeeze()  # Ensure it's 1D
        
        negative_outputs = model(
            input_ids=batch["negative_input_ids"],
            attention_mask=batch["negative_attention_mask"],
            global_attention_mask=batch["negative_global_attention_mask"]
        ).logits.squeeze()  # Ensure it's 1D

        # The target is set to 1, indicating that the positive should be ranked higher than the negative.
        loss = criterion(positive_outputs, negative_outputs, batch["target"])
        return loss
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Extract the inputs for positive samples
        positive_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            global_attention_mask=inputs["global_attention_mask"]
        ).logits.squeeze()  # Ensure it's 1D
        
        negative_outputs = model(
            input_ids=inputs["negative_input_ids"],
            attention_mask=inputs["negative_attention_mask"],
            global_attention_mask=inputs["negative_global_attention_mask"]
        ).logits.squeeze()  # Ensure it's 1D
        return {"positive_logits": positive_outputs, "negative_logits": negative_outputs}

    def evaluation_loop(self, dataloader, description="Evaluation New", prediction_loss_only=False, ignore_keys=None, metric_key_prefix="", **kwargs):
        # Initialization
        model.eval()
        all_positive_logits = []
        all_negative_logits = []
        all_targets = []
        
        with torch.no_grad():
            # Loop through all batches
            for inputs in tqdm(dataloader, desc=description):
                # Get the model predictions (logits for positive and negative samples)
                predictions = self.prediction_step(self.model, inputs, prediction_loss_only)

                # Store the logits and targets for batched loss computation later
                all_positive_logits.append(predictions["positive_logits"])
                all_negative_logits.append(predictions["negative_logits"])
                all_targets.append(inputs["target"])

        # Batched loss computation
        all_positive_logits = torch.cat(all_positive_logits).to(inputs["input_ids"].device)
        all_negative_logits = torch.cat(all_negative_logits).to(inputs["input_ids"].device)
        all_targets = torch.cat(all_targets).to(inputs["input_ids"].device)
        print ("All positive logits are: ", all_positive_logits)
        print ("All negative logits are: ", all_negative_logits)
        print ("All targets are: ", all_targets)
        total_loss = criterion(all_positive_logits, all_negative_logits, all_targets)
        print ("Total loss is: ", total_loss)
        avg_loss = total_loss.item()

        print ("\nAverage loss is: ", avg_loss)

        # Here you can compute other metrics if necessary. In this example, we will just return the average loss.
        model.train()
        return {"eval_loss": avg_loss}


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # Use the custom evaluation loop to get the eval loss
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.evaluation_loop(eval_dataloader)

        # Extract evaluation loss
        eval_loss = output.get("eval_loss", None)

        # Construct the metrics dictionary
        metrics = {
            f"{metric_key_prefix}_loss": eval_loss
            # You can add more metrics here if they are part of the `output`
        }

        # Log, save metrics, and maybe do other post-processing 
        # that the original Trainer does after evaluation
        self.log_metrics("eval", metrics)
        self.save_metrics("eval", metrics)
        self.save_state()

        # Return the metrics in expected format
        return metrics

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)



print ("Training the model now margin augmented new new new error handling new new--- \n")
# Train the model
trainer.train()


trainer.evaluate()