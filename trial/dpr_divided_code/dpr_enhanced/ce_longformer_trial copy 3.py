question = "Your Question Here"
context = "Your Context Here"
input_text = f"[QUESTION] {question} [CONTEXT] {context}"

tokenized_input = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=MAX_SEQ_LEN, truncation=True)
attention_mask = create_global_attention_mask(input_text, tokenizer)


with torch.no_grad():
    model.eval()
    outputs = model(**tokenized_input, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()



question = "Your Question Here"
contexts = ["Context 1", "Context 2", "...", "Context N"]

tokenized_inputs = []
attention_masks = []

for context in contexts:
    input_text = f"[QUESTION] {question} [CONTEXT] {context}"
    
    tokenized_input = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=MAX_SEQ_LEN, truncation=True)
    attention_mask = create_global_attention_mask(input_text, tokenizer)
    
    tokenized_inputs.append(tokenized_input)
    attention_masks.append(attention_mask)



logits_list = []

model.eval()
with torch.no_grad():
    for tokenized_input, attention_mask in zip(tokenized_inputs, attention_masks):
        outputs = model(**tokenized_input, attention_mask=attention_mask)
        logits = outputs.logits[:, 1]  # Assuming class 1 is the "Positive" class
        logits_list.append(logits.item())


# Pairing contexts with their respective logits
context_logit_pairs = list(zip(contexts, logits_list))

# Sorting based on logits (in descending order for highest logits first)
sorted_context_logit_pairs = sorted(context_logit_pairs, key=lambda x: x[1], reverse=True)

# Extracting the ranked contexts
ranked_contexts = [context for context, _ in sorted_context_logit_pairs]
