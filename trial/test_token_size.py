# import os
# from transformers import T5Tokenizer

# def read_sentences_from_file(file_path):
#     with open(file_path, 'r') as f:
#         return [line.strip() for line in f.readlines()]

# def find_max_words_and_tokens(sentences, tokenizer):
#     max_word_count = 0
#     max_token_count = 0
#     max_word_sentence = ''
#     max_token_sentence = ''

#     for sentence in sentences:
#         word_count = len(sentence.split())
#         token_count = len(tokenizer.encode(sentence))

#         if word_count > max_word_count:
#             max_word_count = word_count
#             max_word_sentence = sentence

#         if token_count > max_token_count:
#             max_token_count = token_count
#             max_token_sentence = sentence

#     return max_word_sentence, max_word_count, max_token_sentence, max_token_count

# def main():
#     file_path = '/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/data/mdd_all/dd-generation-structure/train.source'

#     if not os.path.isfile(file_path):
#         print(f"File '{file_path}' not found.")
#         return

#     sentences = read_sentences_from_file(file_path)
#     tokenizer = T5Tokenizer.from_pretrained('t5-base')

#     max_word_sentence, max_word_count, max_token_sentence, max_token_count = find_max_words_and_tokens(sentences, tokenizer)

#     print(f"Max word count sentence: '{max_word_sentence}' with {max_word_count} words.")
#     print(f"Max token count sentence: '{max_token_sentence}' with {max_token_count} tokens.")

# if __name__ == "__main__":
#     main()


import torch
from transformers import T5Tokenizer

t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")

def truncate_question_sequences(question, max_question_len=255):
    question_tokens = t5_tokenizer(question, truncation=True, max_length=max_question_len, return_tensors="pt")
    print(len(question_tokens["input_ids"].squeeze()))
    
    decoded_text = t5_tokenizer.decode(question_tokens["input_ids"].squeeze(), skip_special_tokens=True)
    return decoded_text


def process_question_list(questions, max_question_len=255):
    truncated_questions = [truncate_question_sequences(q, max_question_len) for q in questions]
    return truncated_questions

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of questions
questions = [
    "What is the capital of France? What is the capital of France? What is the capital of France? What is the capital of France?",
    "How does photosynthesis work? How does photosynthesis work? How does photosynthesis work? How does photosynthesis work?",
    "What is the meaning of life? How does photosynthesis work? How does photosynthesis work? How does photosynthesis work?"
]

# Apply the function to the list of questions
truncated_questions = process_question_list(questions, 11)
print ("\n\n\n\n")
print (truncated_questions)

truncated_questions = process_question_list(truncated_questions, 10)
print ("\n\n\n\n")
print (truncated_questions)

truncated_questions = process_question_list(truncated_questions, 10)
print ("\n\n\n\n")
print (truncated_questions)


truncated_questions = process_question_list(truncated_questions, 10)
print ("\n\n\n\n")
print (truncated_questions)


truncated_questions = process_question_list(truncated_questions, 10)
print ("\n\n\n\n")
print (truncated_questions)
# Now you can use truncated_questions_tensors with your PyTorch model
