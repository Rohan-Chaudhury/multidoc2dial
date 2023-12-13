import os
GPUS="0,1"

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
random.seed(42)


config = LongformerConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 15
# LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 1024
EVAL_BATCH_SIZE = 32
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
config.attention_window = 512
config.attention_probs_dropout_prob = 0.25
config.hidden_dropout_prob = 0.25

model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', config=config)

model = model.to(device)

tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096', max_length = 1024)




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
def preprocess_data_for_pairs(training_data, negative_weight=30):
    positive_texts = []
    negative_texts = []

    for item in tqdm(training_data[:100]):
        question = preprocess_question(item["question"])
        positive_ctx = remove_extra_spaces(item["positive_psg"]["text"])

        negative_ctxs = item["negative_psgs"][:negative_weight]
        for negative_ctx in negative_ctxs:
            negative_context = remove_extra_spaces(negative_ctx)
            
            positive_texts.append(f"[QUESTION] {question} </s> [CONTEXT] {positive_ctx}")
            negative_texts.append(f"[QUESTION] {question} </s> [CONTEXT] {negative_context}")

    combined = list(zip(positive_texts, negative_texts))
    random.shuffle(combined)
    positive_texts, negative_texts = zip(*combined)

    return positive_texts, negative_texts

positive_train, negative_train = preprocess_data_for_pairs(training_data, negative_weight=14)
positive_val, negative_val = preprocess_data_for_pairs(validation_data, negative_weight=30)

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


    def __getitem__(self, idx):
        # print ("idx is: ", idx)
        positive_text = self.positive_inputs[idx]
        negative_text = self.negative_inputs[idx]

        positive_tokenized = self.tokenizer(positive_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        positive_global_attention_mask = create_global_attention_mask(positive_text, self.tokenizer)
        negative_tokenized = self.tokenizer(negative_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        negative_global_attention_mask = create_global_attention_mask(negative_text, self.tokenizer)

        return {
            "input_ids": positive_tokenized["input_ids"].squeeze(),
            "attention_mask": positive_tokenized["attention_mask"].squeeze(),
            "global_attention_mask": positive_global_attention_mask,
            "negative_input_ids": negative_tokenized["input_ids"].squeeze(),
            "negative_attention_mask": negative_tokenized["attention_mask"].squeeze(),
            "negative_global_attention_mask": negative_global_attention_mask,
            "target": torch.tensor(1)  # Indicates positive should be ranked higher than negative
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


criterion = nn.MarginRankingLoss(margin=1.0).to(device)


from transformers import Trainer, EvalPrediction
from transformers.trainer_utils import PredictionOutput



from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, batch):
        model.train()
        positive_outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            global_attention_mask=batch["global_attention_mask"].to(device)
        ).logits.squeeze()  # Ensure it's 1D
        
        negative_outputs = model(
            input_ids=batch["negative_input_ids"].to(device),
            attention_mask=batch["negative_attention_mask"].to(device),
            global_attention_mask=batch["negative_global_attention_mask"].to(device)
        ).logits.squeeze()  # Ensure it's 1D

        # The target is set to 1, indicating that the positive should be ranked higher than the negative.
        loss = criterion(positive_outputs, negative_outputs, batch["target"].to(device))
        return loss
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Extract the inputs for positive samples
        positive_outputs = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            global_attention_mask=inputs["global_attention_mask"].to(device)
        ).logits.squeeze()  # Ensure it's 1D
        
        negative_outputs = model(
            input_ids=inputs["negative_input_ids"].to(device),
            attention_mask=inputs["negative_attention_mask"].to(device),
            global_attention_mask=inputs["negative_global_attention_mask"].to(device)
        ).logits.squeeze()  # Ensure it's 1D
        return {"positive_logits": positive_outputs, "negative_logits": negative_outputs}

    def evaluation_loop(self, dataloader, description="Evaluation New", prediction_loss_only=False, ignore_keys=None, metric_key_prefix="", **kwargs):
        # Initialization
        model.eval()
        eval_losses = []
        all_positive_logits = []
        all_negative_logits = []
        with torch.no_grad():
            # Loop through all batches
            for inputs in tqdm(dataloader, desc=description):
                # Get the model predictions (logits for positive and negative samples)
                predictions = self.prediction_step(self.model, inputs, prediction_loss_only)

                # Compute the MarginRankingLoss for this batch
                loss = criterion(predictions["positive_logits"], predictions["negative_logits"], inputs["target"].to(device))
                eval_losses.append(loss.item())

                # If you want to compute other metrics later, save the logits
                # all_positive_logits.extend(predictions["positive_logits"].detach().cpu().numpy())
                # all_negative_logits.extend(predictions["negative_logits"].detach().cpu().numpy())

        # Calculate the average loss
        avg_loss = sum(eval_losses) / len(eval_losses)

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



print ("Training the model now margin 3090--- \n")
# Train the model
trainer.train()


trainer.evaluate()