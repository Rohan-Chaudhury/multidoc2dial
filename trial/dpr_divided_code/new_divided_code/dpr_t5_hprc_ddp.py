import json
import torch

from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import transformers

print(transformers.__version__)
from transformers import T5Tokenizer
import json

from tqdm.auto import tqdm
from transformers import T5EncoderModel

from transformers import T5Config


import torch

from transformers import DPRQuestionEncoder, DPRContextEncoder
from torch.nn.utils import clip_grad_norm_

    
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from math import ceil
from tqdm import tqdm
import copy
import random
import datetime
import os

import os

import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import random
import os
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


import json
import re
from datasets import Dataset

def main():
    config = {
        "epochs": 20,
        "batch_size": 2,
        "learning_rates": {
            "cross_encoder": 1e-5
        },
        "gradient_accumulation_steps": 256,
        "max_length": 1024,
        "patience": 10,
        "temperature": 1.0,
    }

    def kaiming_initialization(module, nonlinearity="leaky_relu", mode="fan_in"):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            # Set the negative slope for leaky_relu that approximates gelu
            negative_slope = 0.01 if nonlinearity == "leaky_relu" else 0
            nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity, a=negative_slope)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


    class T5CrossEncoder(nn.Module):
        def __init__(self, pretrained_model_name, model_max_length, dropout_rate=0.1):
            super().__init__()
            config = T5Config.from_pretrained(pretrained_model_name)
            config.model_max_length = model_max_length
            self.t5 = T5EncoderModel.from_pretrained(pretrained_model_name, config=config)
            self.dropout = nn.Dropout(dropout_rate)
            self.layer_norm = nn.LayerNorm(config.d_model)
            self.classifier = self._init_classifier(config.d_model)

        def _init_classifier(self, d_model):
            classifier = nn.Linear(d_model, 1)
            kaiming_initialization(classifier,  nonlinearity="leaky_relu")
            return classifier

        def forward(self, input_ids, attention_mask):
            outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last_hidden_state = outputs.last_hidden_state
            pooled_output = last_hidden_state[:, 0]
            pooled_output = self.dropout(pooled_output)
            pooled_output = self.layer_norm(pooled_output)
            logits = self.classifier(pooled_output)
            return logits




    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model_max_length = 1024

    checkpoint_path_ce =  "/scratch/user/rohan.chaudhury/t5_large_500/2023-05-03_12-06-33/ce_checkpoint.pth"

    checkpoint_ce = torch.load(checkpoint_path_ce, map_location=device)
    print ("Loss: ", checkpoint_ce['loss'])
    t5_pretrained_model_name = "/scratch/user/rohan.chaudhury/t5_large_model/t5_large_model"
    t5_cross_encoder = T5CrossEncoder(t5_pretrained_model_name, model_max_length)
    t5_cross_encoder= nn.DataParallel(t5_cross_encoder)
    t5_cross_encoder.to(device)
    t5_cross_encoder.load_state_dict(checkpoint_ce['cross_encoder_state_dict'])
    t5_cross_encoder.train()
    print ("\n Checkpoint loaded successfully! \n ")
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_pretrained_model_name, model_max_length=model_max_length)




    if os.path.exists("/scratch/user/rohan.chaudhury/new_divided_code/t5_training_data_flan.json"):
        with open("/scratch/user/rohan.chaudhury/new_divided_code/t5_training_data_flan.json", "r") as f:
            preprocessed_data = json.load(f)



    if os.path.exists("/scratch/user/rohan.chaudhury/new_divided_code/t5_validation_data_flan.json"):
        with open("/scratch/user/rohan.chaudhury/new_divided_code/t5_validation_data_flan.json", "r") as f:
            preprocessed_validation_data = json.load(f)

    # preprocessed_data = {key: value[:1000] for key, value in preprocessed_data.items()}
    # preprocessed_validation_data = {key: value[:1000] for key, value in preprocessed_validation_data.items()}

    train_dataset = Dataset.from_dict(preprocessed_data)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    validation_dataset  = Dataset.from_dict(preprocessed_validation_data)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config["batch_size"], shuffle=True)

    print ("\n\n New Flan Datasets loaded 10\n\n")



    class EarlyStopping:
        def __init__(self, patience=5, delta=0.001):
            self.patience = patience
            self.delta = delta
            self.best_loss = None
            self.counter = 0

        def __call__(self, loss):
            if self.best_loss is None:
                self.best_loss = loss
            elif self.best_loss - loss > self.delta:
                self.counter = 0
                self.best_loss = loss
            else:
                self.counter += 1

            if self.counter >= self.patience:
                return True
            return False


    def train_dpr_model(config, train_dataset, validation_dataset, cross_encoder, t5_tokenizer, device):
        epochs = config["epochs"]
        batch_size = config["batch_size"]
        learning_rates = config["learning_rates"]
        gradient_accumulation_steps = config["gradient_accumulation_steps"]

        max_length = config["max_length"]
        patience = config["patience"]

        weight_decay = 1e-5

        adam_epsilon = 1e-8
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_cross_encoder_parameters = [
            {
                "params": [p for n, p in cross_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in cross_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        early_stopping = EarlyStopping(patience=patience)
        cross_encoder_optimizer = AdamW(optimizer_grouped_cross_encoder_parameters, lr=learning_rates['cross_encoder'], eps=adam_epsilon)
        
        number_of_batches = ceil(len(train_dataset) / batch_size)
        total_steps = number_of_batches * epochs // gradient_accumulation_steps

        base_learning_rate = learning_rates['cross_encoder']

        T_0 = 2000  # Number of iterations for the first restart
        T_mult = 2  # Multiplicative factor to increase the cycle length after each restart

        cross_encoder_scheduler = CosineAnnealingWarmRestarts(cross_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)

        # model.to(device)
        best_val_loss = float('inf')
        best_model = None

    

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            total_loss = 0.0

            train_iter=tqdm(train_dataloader, desc="Training", ncols=100)
            for idx, batch in enumerate(train_iter):
                

                questions = batch["question"]
                positive_contexts = batch["positive_context"]
                negative_contexts = batch["negative_context"]

                positive_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, positive_contexts)]
                negative_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, negative_contexts)]
                t5_input = positive_t5_input + negative_t5_input

                t5_encodings = t5_tokenizer(t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
                t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)

                cross_encoder_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()
                cross_encoder_labels = torch.tensor([1] * len(questions) + [0] * len(questions), dtype=torch.float, device=device)
                cross_encoder_loss = F.binary_cross_entropy_with_logits(cross_encoder_logits, cross_encoder_labels)



                # Compute the combined loss using updated weights
                combined_loss =  cross_encoder_loss 
                total_loss += combined_loss.item()



                combined_loss = combined_loss / gradient_accumulation_steps
                combined_loss.backward()

                if (idx + 1) % gradient_accumulation_steps == 0  or (idx + 1 == len(train_dataloader)):
                    # Apply gradient clipping
                    clip_grad_norm_(cross_encoder.parameters(), max_norm=1.0)

                    cross_encoder_optimizer.step()


                    cross_encoder_scheduler.step()


                    cross_encoder_optimizer.zero_grad()

                
                train_iter.set_description(f"Training (loss = {cross_encoder_loss.item():.4f})")
                train_iter.refresh()

            avg_train_loss = total_loss / number_of_batches
            print(f"Training loss: {avg_train_loss}")

            # Validation
            cross_encoder.eval()

            ###################################################
            train_cross_encoder_state_dict = copy.deepcopy(cross_encoder.state_dict())
            # Get the current date and time as a string
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            directory_to_save_train="/scratch/user/rohan.chaudhury/t5_large_500/output/train/t5_large_500/"+timestamp
            # Create a new directory with the timestamp
            os.makedirs(directory_to_save_train, exist_ok=True)

            # Save the model checkpoint with additional metadata
            torch.save({
                'epoch': epoch,
                'cross_encoder_state_dict': train_cross_encoder_state_dict,
                'cross_encoder_optimizer_state_dict': cross_encoder_optimizer.state_dict(),
                'loss': avg_train_loss,
            }, os.path.join(directory_to_save_train, "ce_checkpoint.pth"))

            print(f"model checkpoint saved with training loss: {avg_train_loss} in directory {directory_to_save_train}")

            #######################################
            del train_cross_encoder_state_dict
            torch.cuda.empty_cache()


            total_val_loss = 0.0
            number_of_batches_validation = ceil(len(validation_dataset) / batch_size)

            val_iter=tqdm(validation_dataloader, desc="Validation", ncols=100)
            for batch in val_iter:
                questions = batch["question"]
                positive_contexts = batch["positive_context"]
                negative_contexts = batch["negative_context"]

                    


                positive_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, positive_contexts)]
                negative_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, negative_contexts)]
                t5_input = positive_t5_input + negative_t5_input

                t5_encodings = t5_tokenizer(t5_input, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
                t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)

                with torch.no_grad():
                    cross_encoder_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()
                    cross_encoder_labels = torch.tensor([1] * len(questions) + [0] * len(questions), dtype=torch.float, device=device)
                    cross_encoder_loss = F.binary_cross_entropy_with_logits(cross_encoder_logits, cross_encoder_labels)


                # Combine the losses
                combined_loss = cross_encoder_loss 


                total_val_loss += combined_loss.item()
                val_iter.set_description(f"Validation (loss = {cross_encoder_loss.item():.4f})")
                val_iter.refresh()
                
            avg_val_loss = total_val_loss / number_of_batches_validation
            print(f"Validation loss: {avg_val_loss}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_cross_encoder_state_dict = copy.deepcopy(cross_encoder.state_dict())
                # Get the current date and time as a string
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                directory_to_save="/scratch/user/rohan.chaudhury/t5_large_500/output/t5_large_500/"+timestamp
                # Create a new directory with the timestamp
                os.makedirs(directory_to_save, exist_ok=True)

                # Save the model checkpoint with additional metadata
                torch.save({
                    'epoch': epoch,
                    'cross_encoder_state_dict': best_cross_encoder_state_dict,
                    'cross_encoder_optimizer_state_dict': cross_encoder_optimizer.state_dict(),
                    'loss': avg_val_loss,
                }, os.path.join(directory_to_save, "ce_checkpoint.pth"))

                print(f"Best model checkpoint saved with validation loss: {best_val_loss} in directory {directory_to_save}")

                print(f"Best model saved with validation loss: {best_val_loss}")

                del best_cross_encoder_state_dict
                torch.cuda.empty_cache()
            # else:
            #     num_epochs_without_improvement += 1

            # if num_epochs_without_improvement >= patience:
            #     print(f"Early stopping triggered. No improvement in validation loss for {patience} consecutive epochs.")
            #     break
            if early_stopping(avg_val_loss):
                print("Early stopping triggered")
                break

        print("Training complete.")
        return best_model



    # Calculate total training steps and set num_warmup_steps as a fraction of total steps
    total_steps = (len(train_dataset) // (config["batch_size"] * config["gradient_accumulation_steps"])) * config["epochs"]
    config["num_warmup_steps"] = int(0.1 * total_steps)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    best_model = train_dpr_model(config, train_dataset, validation_dataset, t5_cross_encoder, t5_tokenizer, device)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Number of GPUs per node")
    args = parser.parse_args()

    world_size = args.nodes * args.gpus_per_node
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(main, nprocs=args.gpus_per_node, args=(args.gpus_per_node, world_size))