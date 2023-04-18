

class T5CrossEncoder(nn.Module):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.t5 = T5EncoderModel.from_pretrained(pretrained_model_name)
        self.classifier = nn.Linear(self.t5.config.d_model, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits


class CustomDPRQuestionEncoderWithDropout(nn.Module):
    def __init__(self, model_name, dropout_rate):
        super(CustomDPRQuestionEncoderWithDropout, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return  self.linear(self.dropout(self.layer_norm(pooled_output)))
        # return self.linear(pooled_output)
        # return outputs.pooler_output

def compute_scores(question_embeddings, context_embeddings):
    return torch.matmul(question_embeddings, context_embeddings.t())

class CustomDPRContextEncoder(nn.Module):
    def __init__(self, model_name, dropout_rate):
        super(CustomDPRContextEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        self.linear = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return  self.linear(self.dropout(self.layer_norm(pooled_output)))

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_question(question):
    turns = question.split("[SEP]")
    questions=turns[0]
    turns=[turns[1]]
    turns = [turn.strip() for turn in turns if turn.strip()]
    turns = [turn.split("||") for turn in turns]
    turns = [turn[::-1] for turn in turns]  # Reverse the order of previous turns
    turns = [" || ".join(turn) for turn in turns]

    return "Query: "+ questions.lower()+ " || Context: "+  " ".join(turns).lower() 

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
            all_negative_ctxs = (negative_ctxs * negative_weight) + (hard_negative_ctxs * hard_negative_weight)

            for negative_ctx in all_negative_ctxs:
                negative_context = remove_extra_spaces(negative_ctx["title"].lower() + " " + negative_ctx["text"].lower())

                train_data["question"].append(question)
                train_data["positive_context"].append(positive_context)
                train_data["negative_context"].append(negative_context)

    return train_data


def contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings, margin=1.0):
    pos_distances = torch.norm(anchor_embeddings - positive_embeddings, dim=-1)
    neg_distances = torch.norm(anchor_embeddings - negative_embeddings, dim=-1)
    return F.relu(pos_distances - neg_distances + margin).mean()

class DPRCombinedModel(nn.Module):
    def __init__(self, question_encoder: DPRQuestionEncoder, context_encoder: DPRContextEncoder):
        super(DPRCombinedModel, self).__init__()
        self.question_encoder = question_encoder
        self.context_encoder = context_encoder

    def forward(self, question_input_ids, question_attention_mask, context_input_ids, context_attention_mask):
        question_outputs = self.question_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask)
        context_outputs = self.context_encoder(input_ids=context_input_ids, attention_mask=context_attention_mask)
        return question_outputs, context_outputs
    


def process_batch(batch, question_tokenizer, context_tokenizer, max_length, device):
    questions = batch["question"]
    positive_contexts = batch["positive_context"]
    negative_contexts = batch["negative_context"]

    anchor_encodings = question_tokenizer(questions, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    positive_encodings = context_tokenizer(positive_contexts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    negative_encodings = context_tokenizer(negative_contexts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

    anchor_input_ids, anchor_attention_mask = anchor_encodings['input_ids'].to(device), anchor_encodings['attention_mask'].to(device)
    positive_input_ids, positive_attention_mask = positive_encodings['input_ids'].to(device), positive_encodings['attention_mask'].to(device)
    negative_input_ids, negative_attention_mask = negative_encodings['input_ids'].to(device), negative_encodings['attention_mask'].to(device)

    return anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask


def train_dpr_model(config, train_dataset, validation_dataset, model, cross_encoder, question_tokenizer, context_tokenizer, t5_tokenizer, device):
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rates = config["learning_rates"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    num_warmup_steps = config["num_warmup_steps"]
    max_length = config["max_length"]
    patience = config["patience"]
    temperature = config["temperature"]

    weight_decay = 1e-5

    adam_epsilon = 1e-8
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_question_encoder_parameters = [
        {
            "params": [p for n, p in model.question_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.question_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer_grouped_context_encoder_parameters = [
        {
            "params": [p for n, p in model.context_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.context_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer_grouped_cross_encoder_parameters = [
        {
            "params": [p for n, p in cross_encoder.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in cross_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    early_stopping = EarlyStopping(patience=patience)
    cross_encoder_optimizer = AdamW(optimizer_grouped_cross_encoder_parameters, lr=learning_rates['cross_encoder'], eps=adam_epsilon)
    
    question_encoder_optimizer = AdamW(optimizer_grouped_question_encoder_parameters , lr=learning_rates['question_encoder'],  eps=adam_epsilon)
    context_encoder_optimizer = AdamW(optimizer_grouped_context_encoder_parameters, lr=learning_rates['context_encoder'],  eps=adam_epsilon)
    number_of_batches = ceil(len(train_dataset) / batch_size)
    total_steps = number_of_batches * epochs // gradient_accumulation_steps

    base_learning_rate = 1e-5

    T_0 = 2000  # Number of iterations for the first restart
    T_mult = 2  # Multiplicative factor to increase the cycle length after each restart

    question_encoder_scheduler = CosineAnnealingWarmRestarts(question_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)

    context_encoder_scheduler = CosineAnnealingWarmRestarts(context_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)

    cross_encoder_scheduler = CosineAnnealingWarmRestarts(cross_encoder_optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_learning_rate)

    # model.to(device)
    best_val_loss = float('inf')
    best_model = None

    # Initialize loss weights
    contrastive_weight = 1 / 3
    cross_encoder_weight = 1 / 3
    dr_weight = 1 / 3

    # Initialize running averages for losses
    running_contrastive_loss = 0.0
    running_cross_encoder_loss = 0.0
    running_dr_loss = 0.0

    # Define the smoothing factor for updating the running averages
    smoothing_factor = 0.1

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_loss = 0.0

        alpha = 0.2
        train_iter=tqdm(train_dataloader, desc="Training", ncols=100)
        for idx, batch in enumerate(train_iter):
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = process_batch(batch, question_tokenizer, context_tokenizer, max_length, device)
            # example_indices = shuffled_indices[i:i + batch_size]

            questions = batch["question"]
            positive_contexts = batch["positive_context"]
            negative_contexts = batch["negative_context"]

            # Compute embeddings
            anchor_embeddings = model.question_encoder(anchor_input_ids, anchor_attention_mask)
            positive_embeddings = model.context_encoder(positive_input_ids, positive_attention_mask)
            negative_embeddings = model.context_encoder(negative_input_ids, negative_attention_mask)

            # Compute contrastive loss
            loss = contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            positive_scores = compute_scores(anchor_embeddings, positive_embeddings)
            negative_scores = compute_scores(anchor_embeddings, negative_embeddings)
            margin = 1.0
            dr_loss = torch.clamp(margin - positive_scores + negative_scores, min=0).mean()
            # Prepare cross-encoder inputs
            positive_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, positive_contexts)]
            negative_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, negative_contexts)]
            t5_input = positive_t5_input + negative_t5_input

            t5_encodings = t5_tokenizer(t5_input, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)

            cross_encoder_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()
            cross_encoder_labels = torch.tensor([1] * len(questions) + [0] * len(questions), dtype=torch.float, device=device)
            cross_encoder_loss = F.binary_cross_entropy_with_logits(cross_encoder_logits, cross_encoder_labels)


            # Update running averages for losses
            running_contrastive_loss = (1 - smoothing_factor) * running_contrastive_loss + smoothing_factor * loss.item()
            running_cross_encoder_loss = (1 - smoothing_factor) * running_cross_encoder_loss + smoothing_factor * cross_encoder_loss.item()
            running_dr_loss = (1 - smoothing_factor) * running_dr_loss + smoothing_factor * dr_loss.item()

            # Update weights based on the running averages of losses
            total_running_loss = running_contrastive_loss + running_cross_encoder_loss + running_dr_loss
            contrastive_weight = running_contrastive_loss / total_running_loss
            cross_encoder_weight = running_cross_encoder_loss / total_running_loss
            dr_weight = running_dr_loss / total_running_loss

            combined_loss = contrastive_weight * loss + cross_encoder_weight * cross_encoder_loss + dr_weight * dr_loss
            total_loss += combined_loss.item()
            combined_loss = combined_loss / gradient_accumulation_steps
            combined_loss.backward()

            if (idx + 1) % gradient_accumulation_steps == 0:
                # Apply gradient clipping
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                clip_grad_norm_(cross_encoder.parameters(), max_norm=1.0)

                question_encoder_optimizer.step()
                context_encoder_optimizer.step()
                cross_encoder_optimizer.step()

                question_encoder_scheduler.step()
                context_encoder_scheduler.step()
                cross_encoder_scheduler.step()

                model.zero_grad()
                cross_encoder.zero_grad()

            
            train_iter.set_description(f"Training (loss = {loss.item():.4f})")
            train_iter.refresh()

        avg_train_loss = total_loss / number_of_batches
        print(f"Training loss: {avg_train_loss}")

        model.eval()
        total_val_loss = 0.0
        number_of_batches_validation = ceil(len(validation_dataset) / batch_size)

        val_iter=tqdm(validation_dataloader, desc="Validation", ncols=100)
        for batch in val_iter:
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = process_batch(batch, question_tokenizer, context_tokenizer, max_length, device)
            questions = batch["question"]
            positive_contexts = batch["positive_context"]
            negative_contexts = batch["negative_context"]
            with torch.no_grad():
                # Compute embeddings
                anchor_embeddings = model.question_encoder(anchor_input_ids, anchor_attention_mask)
                positive_embeddings = model.context_encoder(positive_input_ids, positive_attention_mask)
                negative_embeddings = model.context_encoder(negative_input_ids, negative_attention_mask)
                positive_scores = compute_scores(anchor_embeddings, positive_embeddings)
                negative_scores = compute_scores(anchor_embeddings, negative_embeddings)
                
                margin = 1.0
                dr_loss = torch.clamp(margin - positive_scores + negative_scores, min=0).mean()

                # Compute contrastive loss
                loss = contrastive_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

            # Prepare cross-encoder inputs
            positive_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, positive_contexts)]
            negative_t5_input = [f"{q} </s> {c}" for q, c in zip(questions, negative_contexts)]
            t5_input = positive_t5_input + negative_t5_input

            t5_encodings = t5_tokenizer(t5_input, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            t5_input_ids, t5_attention_mask = t5_encodings["input_ids"].to(device), t5_encodings["attention_mask"].to(device)

            with torch.no_grad():
                cross_encoder_logits = cross_encoder(t5_input_ids, t5_attention_mask).squeeze()
                cross_encoder_labels = torch.tensor([1] * len(questions) + [0] * len(questions), dtype=torch.float, device=device)
                cross_encoder_loss = F.binary_cross_entropy_with_logits(cross_encoder_logits, cross_encoder_labels)


            # Combine the losses
            combined_loss = contrastive_weight * loss + cross_encoder_weight * cross_encoder_loss + dr_weight * dr_loss


            total_val_loss += combined_loss.item()
            val_iter.set_description(f"Validation (loss = {loss.item():.4f})")
            val_iter.refresh()
            
        avg_val_loss = total_val_loss / number_of_batches_validation
        print(f"Validation loss: {avg_val_loss}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_cross_encoder_state_dict = copy.deepcopy(cross_encoder.state_dict())
            # Get the current date and time as a string
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            directory_to_save="/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/trial/output/contrastive_discriminative/"+timestamp
            # Create a new directory with the timestamp
            os.makedirs(directory_to_save, exist_ok=True)

            # Save the model checkpoint with additional metadata
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state_dict,
                'cross_encoder_state_dict': best_cross_encoder_state_dict,
                'question_encoder_optimizer_state_dict': question_encoder_optimizer.state_dict(),
                'context_encoder_optimizer_state_dict': context_encoder_optimizer.state_dict(),
                'cross_encoder_optimizer_state_dict': cross_encoder_optimizer.state_dict(),
                'question_encoder_scheduler_state_dict': question_encoder_scheduler.state_dict(),
                'context_encoder_scheduler_state_dict': context_encoder_scheduler.state_dict(),
                'loss': avg_val_loss,
            }, os.path.join(directory_to_save, "checkpoint.pth"))

            print(f"Best model checkpoint saved with validation loss: {best_val_loss} in directory {directory_to_save}")

            print(f"Best model saved with validation loss: {best_val_loss}")

        if early_stopping(avg_val_loss):
            print("Early stopping triggered")
            break

    print("Training complete.")
    return best_model

