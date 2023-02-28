"""Tokenizes the sentences with BertTokenizer as tokenisation costs some time.
"""
import sys
from transformers import AutoTokenizer
import json
from sklearn.model_selection import train_test_split
# BERT_VOCAB = "nlpaueb/legal-bert-base-uncased"
BERT_VOCAB = "t5-large"
# BERT_VOCAB = "bert-base-uncased"
from sentence_transformers import SentenceTransformer
MAX_SEQ_LENGTH = 128


def clean_text(text):
    """
    Clean the given text.

    :param text: input text
    :type text: str
    :return: cleaned string[]
    """
    return text.strip()



def write_in_hsln_format(input,hsln_format_txt_dirpath,tokenizer):


    final_string = ''
    for doc_idx1 in input['doc_data']:
        for doc_idx2 in input['doc_data'][doc_idx1]:
            file_name= doc_idx1+'_'+doc_idx2
            file_name= file_name.replace("\s+", "_")
            file_name= file_name.replace(" ", "_")
            file_name = file_name.replace("\t", "_")
            final_string = final_string + '###' + str(file_name) + "\n"
            for doc_idx3 in input['doc_data'][doc_idx1]\
                                            [doc_idx2]['spans']:
                sentence_txt = clean_text(input['doc_data']\
                                                    [doc_idx1][doc_idx2]['spans']\
                                                    [doc_idx3]['text_sp'])
                sentence_label=doc_idx2
                sentence_label= sentence_label.replace("\s+", "_")
                sentence_label= sentence_label.replace(" ", "_")
                sentence_label = sentence_label.replace("\t", "_")
                sentence_txt = sentence_txt.replace("\r", "")
                if sentence_txt.strip() != "":
                    sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=256)
                    sent_tokens = [str(i) for i in sent_tokens]
                    sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + sentence_label + "\t" + sent_tokens_txt + "\n"
            final_string = final_string + "\n"
    with open(hsln_format_txt_dirpath , "w+") as file:
        file.write(final_string)

# doc_sentence_test = []
# doc_label_test = []
# for doc_idx1 in multidoc2dial_dial_train['dial_data']:
#     for dial in multidoc2dial_dial_train['dial_data'][doc_idx1]:
#         for turns in dial['turns']:
#             if turns['role'] == "user":
#                 doc_sentence_test.append(turns['utterance'])
#                 doc_label_test.append(turns['references'][0]['doc_id'])


def write_in_hsln_format_test(input,hsln_format_txt_dirpath,tokenizer):


    final_string = ''
    for doc_idx1 in input['dial_data']:
        for dial in input['dial_data'][doc_idx1]:
            file_name= dial["dial_id"]
            file_name= file_name.replace("\s+", "_")
            file_name= file_name.replace(" ", "_")
            file_name = file_name.replace("\t", "_")
            final_string = final_string + '###' + str(file_name) + "\n"

            for turns in dial['turns']:
                if turns['role'] == "user":
                    sentence_txt = turns['utterance']
                    sentence_label = turns['references'][0]['doc_id']
                    sentence_label= sentence_label.replace("\s+", "_")
                    sentence_label= sentence_label.replace(" ", "_")
                    sentence_label = sentence_label.replace("\t", "_")
                    sentence_txt = sentence_txt.replace("\r", "")
                    if sentence_txt.strip() != "":
                        sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=256)
                        sent_tokens = [str(i) for i in sent_tokens]
                        sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + sentence_label + "\t" + sent_tokens_txt + "\n"
            final_string = final_string + "\n"
    with open(hsln_format_txt_dirpath , "w+") as file:
        file.write(final_string)

def write_in_hsln_format_train_append(input,hsln_format_txt_dirpath,tokenizer):


    final_string = ''
    for doc_idx1 in input['dial_data']:
        for dial in input['dial_data'][doc_idx1]:
            file_name= dial["dial_id"]
            file_name= file_name.replace("\s+", "_")
            file_name= file_name.replace(" ", "_")
            file_name = file_name.replace("\t", "_")
            final_string = final_string + '###' + str(file_name) + "\n"

            for turns in dial['turns']:
                if turns['role'] == "user":
                    sentence_txt = turns['utterance']
                    sentence_label = turns['references'][0]['doc_id']
                    sentence_label= sentence_label.replace("\s+", "_")
                    sentence_label= sentence_label.replace(" ", "_")
                    sentence_label = sentence_label.replace("\t", "_")
                    sentence_txt = sentence_txt.replace("\r", "")
                    if sentence_txt.strip() != "":
                        sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=256)
                        sent_tokens = [str(i) for i in sent_tokens]
                        sent_tokens_txt = " ".join(sent_tokens)
                final_string = final_string + sentence_label + "\t" + sent_tokens_txt + "\n"
            final_string = final_string + "\n"
    with open(hsln_format_txt_dirpath , "a+") as file:
        file.write(final_string)

def tokenize():
    # [_, train_input_json,dev_input_json,test_input_json] = sys.argv
    train_input_json = '../../data/multidoc2dial/multidoc2dial_doc.json'
    train_extra= '../../data/multidoc2dial/multidoc2dial_dial_train.json'
    dev_input_json = '../../data/multidoc2dial/multidoc2dial_dial_validation.json'
    test_input_json = '../../data/multidoc2dial/multidoc2dial_dial_test.json'
    tokenizer = AutoTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    train_json_format = json.load(open(train_input_json))
    train_extra_json_format = json.load(open(train_extra))
    dev_json_format = json.load(open(dev_input_json))
    test_json_format = json.load(open(test_input_json))
  
    write_in_hsln_format(train_json_format,'datasets/multidoc2dial/train_scibert.txt',tokenizer)
    # write_in_hsln_format_test(train_extra_json_format,'datasets/multidoc2dial/train_scibert.txt',tokenizer)
    write_in_hsln_format_test(dev_json_format, 'datasets/multidoc2dial/dev_scibert.txt', tokenizer)
    write_in_hsln_format_test(test_json_format, 'datasets/multidoc2dial/test_scibert.txt', tokenizer)





tokenize()
