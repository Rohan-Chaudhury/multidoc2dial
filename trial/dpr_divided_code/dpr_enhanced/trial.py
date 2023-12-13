from transformers import LongformerTokenizer

# Load the tokenizer
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

# Get the tokens
cls_token = tokenizer.cls_token
sep_token = tokenizer.sep_token
tokenizer_sep_token_id = tokenizer.sep_token_id

print("Tokenizer sep token id:", tokenizer_sep_token_id)
print ("Tokenizer cls token id:", tokenizer.cls_token_id)


print("CLS Token:", cls_token)
print("SEP Token:", sep_token)






import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerConfig

class DropConnect(nn.Module):
    def __init__(self, drop_prob):
        super(DropConnect, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.rand(x.size()) < keep_prob
        if x.is_cuda:
            mask = mask.to(device=x.device)
        return mask * x / keep_prob

class CustomModel(nn.Module):
    def __init__(self, config):
        super(CustomModel, self).__init__()

        self.longformer = LongformerModel(config)

        # Custom Positional Encoding
        self.positional_encodings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Attention-based pooling
        self.attention_weights = nn.Parameter(torch.randn(config.hidden_size))
        self.softmax = nn.Softmax(dim=1)

        # Classifier with DropConnect
        self.drop_connect = DropConnect(0.5)  # 0.5 is the drop probability for DropConnect. Adjust as needed.
        self.classifier = nn.Linear(config.hidden_size, 2)  # 2 classes, adjust as necessary

    def forward(self, input_ids, attention_mask=None):
        # Pass through Longformer
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]

        # Add custom positional encodings
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        position_embeddings = self.positional_encodings(positions)
        last_hidden_state += position_embeddings
        
        # Self Attention-based pooling
        attention_scores = torch.matmul(last_hidden_state, self.attention_weights)
        attention_probs = self.softmax(attention_scores)
        pooled_output = torch.sum(last_hidden_state * attention_probs.unsqueeze(-1), dim=1)
        
        # Classifier with DropConnect
        logits = self.classifier(self.drop_connect(pooled_output))

        return logits

# Example usage:
config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
model = CustomModel(config)












import torch.nn.functional as F

class CustomModel(nn.Module):
    # ... [rest of the CustomModel definition]

    def fgsm_attack(self, inputs, epsilon, data_grad):
        """
        FGSM adversarial attack.
        """
        sign_data_grad = data_grad.sign()
        perturbed_inputs = inputs + epsilon * sign_data_grad
        return perturbed_inputs

    def forward(self, input_ids, attention_mask=None, adversarial=False):
        # Pass through Longformer
        outputs = self.longformer(input_ids, attention_mask=attention_mask)
        embeddings = outputs[0]

        # If in training mode and adversarial flag is True, apply FGSM attack
        if adversarial and self.training:
            embeddings.requires_grad = True
            logits = self.classifier(embeddings.mean(dim=1))
            loss = F.cross_entropy(logits, targets) # assuming `targets` are your ground truth labels
            self.zero_grad()
            loss.backward()
            data_grad = embeddings.grad.data
            embeddings = self.fgsm_attack(embeddings, epsilon=0.01, data_grad=data_grad)  # epsilon is the perturbation amount

        # Add custom positional encodings
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).to(input_ids.device)
        position_embeddings = self.positional_encodings(positions)
        embeddings += position_embeddings
        
        # Self Attention-based pooling
        attention_scores = torch.matmul(embeddings, self.attention_weights)
        attention_probs = self.softmax(attention_scores)
        pooled_output = torch.sum(embeddings * attention_probs.unsqueeze(-1), dim=1)
        
        # Classifier with DropConnect
        logits = self.classifier(self.drop_connect(pooled_output))

        return logits



# During training:
logits = model(input_ids, attention_mask, adversarial=True)




def forward(self, input_ids, attention_mask, global_attention_mask):
    outputs = self.longformer(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        global_attention_mask=global_attention_mask, 
        return_dict=True
    )

    last_hidden_state = outputs.last_hidden_state

    # Extract the [CLS] representation
    cls_representation = last_hidden_state[:, 0, :]

    # Extract representations of tokens with global attention
    # Note: global_attention_mask should be of shape (batch_size, sequence_length) with 1s for globally attended tokens and 0s otherwise.
    global_attention_representations = last_hidden_state * global_attention_mask.unsqueeze(-1)
    
    # You could take the mean of the representations of globally attended tokens, but let's keep them separate for now:
    summed_global_attention = torch.sum(global_attention_representations, dim=1)
    count_global_attention = global_attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
    mean_global_attention = summed_global_attention / count_global_attention

    # Concatenation:
    combined_representation_concat = torch.cat([cls_representation, mean_global_attention], dim=-1)

    # Averaging:
    combined_representation_avg = (cls_representation + mean_global_attention) / 2

    # Choose the combined representation you want to use:
    final_representation = combined_representation_avg  # or combined_representation_concat

    final_representation = self.dropout(final_representation)
    final_representation = self.layer_norm(final_representation)
    logits = self.classifier(final_representation)

    return logits





class LongformerCrossEncoder(nn.Module):
    def __init__(self, pretrained_model_name, dropout_rate=0.2, pooling_method="attention"):
        super().__init__()
        config = LongformerConfig.from_pretrained(pretrained_model_name)
        config.attention_window = [256] * config.num_hidden_layers
        config.attention_mode = 'sliding_chunks'
        self.longformer = LongformerModel.from_pretrained(pretrained_model_name, config=config)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention_weights = nn.Parameter(torch.randn(config.hidden_size))
        self.classifier = self._init_classifier(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.pooling_method = pooling_method  # Added pooling method parameter

    def _init_classifier(self, hidden_size):
        classifier = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        for module in classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        return classifier

    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask, 
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state

        # Pooling based on the selected method
        if self.pooling_method == "attention":
            attention_scores = torch.matmul(last_hidden_state, self.attention_weights)
            attention_probs = nn.Softmax(dim=1)(attention_scores)
            pooled_output = torch.sum(last_hidden_state * attention_probs.unsqueeze(-1), dim=1)
        elif self.pooling_method == "mean":
            pooled_output = torch.mean(last_hidden_state, dim=1)
        elif self.pooling_method == "max":
            pooled_output = torch.max(last_hidden_state, dim=1)[0]
        elif self.pooling_method == "cls":
            pooled_output = last_hidden_state[:, 0, :]
        else:
            raise ValueError("Invalid pooling method")

        pooled_output = self.dropout(pooled_output)
        pooled_output = self.layer_norm(pooled_output)  # Uncommented the LayerNorm
        logits = self.classifier(pooled_output)

        return torch.sigmoid(logits)  # Added sigmoid activation for output








import torch.nn as nn
from transformers import LongformerModel, LongformerConfig

class LongformerCrossEncoder(nn.Module):
    def __init__(self, pretrained_model_name, dropout_rate=0.2):
        super().__init__()
        config = LongformerConfig.from_pretrained(pretrained_model_name)
        config.attention_window = [256] * config.num_hidden_layers
        config.attention_mode = 'sliding_chunks'
        self.longformer = LongformerModel.from_pretrained(pretrained_model_name, config=config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = self._init_classifier(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Learnable weights for weighted averaging
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def _init_classifier(self, hidden_size):
        classifier = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        for module in classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        return classifier

    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask, 
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state
        cls_representation = last_hidden_state[:, 0, :]

        global_attention_representations = last_hidden_state * global_attention_mask.unsqueeze(-1)
        summed_global_attention = torch.sum(global_attention_representations, dim=1)
        count_global_attention = global_attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        mean_global_attention = summed_global_attention / count_global_attention

        # Weighted Averaging
        weights = nn.Softmax(dim=0)(torch.stack([self.alpha, self.beta]))
        combined_representation = (weights[0] * cls_representation + weights[1] * mean_global_attention)

        combined_representation = self.dropout(combined_representation)
        combined_representation = self.layer_norm(combined_representation)
        logits = self.classifier(combined_representation)

        return logits

# Instantiate the model
model = LongformerCrossEncoder(pretrained_model_name="allenai/longformer-base-4096")








import torch.nn as nn
from transformers import LongformerModel, LongformerConfig

class LongformerCrossEncoder(nn.Module):
    def __init__(self, pretrained_model_name, dropout_rate=0.2):
        super().__init__()
        config = LongformerConfig.from_pretrained(pretrained_model_name)
        config.attention_window = [256] * config.num_hidden_layers
        config.attention_mode = 'sliding_chunks'
        self.longformer = LongformerModel.from_pretrained(pretrained_model_name, config=config)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = self._init_classifier(2 * config.hidden_size)  # Adjusting for concatenation
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size)  # Adjusting for concatenation

    def _init_classifier(self, hidden_size):
        classifier = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        for module in classifier:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        return classifier

    def forward(self, input_ids, attention_mask, global_attention_mask):
        outputs = self.longformer(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            global_attention_mask=global_attention_mask, 
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state
        cls_representation = last_hidden_state[:, 0, :]

        global_attention_representations = last_hidden_state * global_attention_mask.unsqueeze(-1)
        summed_global_attention = torch.sum(global_attention_representations, dim=1)
        count_global_attention = global_attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
        mean_global_attention = summed_global_attention / count_global_attention

        # Concatenate
        combined_representation = torch.cat([cls_representation, mean_global_attention], dim=-1)

        combined_representation = self.dropout(combined_representation)
        combined_representation = self.layer_norm(combined_representation)
        logits = self.classifier(combined_representation)

        return logits

# Instantiate the model
model = LongformerCrossEncoder(pretrained_model_name="allenai/longformer-base-4096")
