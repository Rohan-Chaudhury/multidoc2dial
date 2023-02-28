# %%
import json
import sys
import os
from datasets import load_dataset

_PATH = "/home/grads/r/rohan.chaudhury/multidoc2dial/multidoc2dial/"
sys.path.append(os.path.join(_PATH))  # noqa: E402 # isort:skip

# %%
val_data = json.load(open(_PATH + 'data/multidoc2dial/multidoc2dial_dial_validation.json','r'))
docs = json.load(open(_PATH + 'data/multidoc2dial/multidoc2dial_doc.json','r'))

DAs      = [line.strip() for line in open(_PATH + 'data/mdd_all/dd-generation-structure/val.{}'.format('da'), "r").readlines()]
domains  = [line.strip() for line in open(_PATH + 'data/mdd_all/dd-generation-structure/val.{}'.format('domain'), "r").readlines()]
pids     = [line.strip() for line in open(_PATH + 'data/mdd_all/dd-generation-structure/val.{}'.format('pids'), "r").readlines()]
qids     = [line.strip() for line in open(_PATH + 'data/mdd_all/dd-generation-structure/val.{}'.format('qids'), "r").readlines()]
sources  = [line.strip() for line in open(_PATH + 'data/mdd_all/dd-generation-structure/val.{}'.format('source'), "r").readlines()]
targets  = [line.strip() for line in open(_PATH + 'data/mdd_all/dd-generation-structure/val.{}'.format('target'), "r").readlines()]
titles   = [line.strip() for line in open(_PATH + 'data/mdd_all/dd-generation-structure/val.{}'.format('titles'), "r").readlines()]

# %%
def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()

def split_text_section(spans, title):
    def get_text(buff, title, span):
        text = " ".join(buff).replace("\n", " ")
        parent_titles = [title.replace("/", "-").rsplit("#")[0]]
        if len(span["parent_titles"]) > 1:
            parent_titles = [ele['text'].replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]]
        text = " / ".join(parent_titles) + " // " + text
        return text2line(text)

    buff = []
    pre_sec, pre_title, pre_span = None, None, None
    passages = []
    subtitles = []
        
    for span_id in spans:
        span = spans[span_id]
        parent_titles = title
        if len(span["parent_titles"]) > 1:                        
            parent_titles = [ele['text'].replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]]
            parent_titles = " / ".join(parent_titles)
        if pre_sec == span["id_sec"] or pre_title == span["title"].strip():
            buff.append(span["text_sp"])
        elif buff:
            text = get_text(buff, title, pre_span)
            passages.append(text)
            subtitles.append(parent_titles)
            buff = [span["text_sp"]]
        else:
            buff.append(span["text_sp"])
        pre_sec = span["id_sec"]
        pre_span = span
        pre_title = span["title"].strip()
    if buff:
        text = get_text(buff, title, span)
        passages.append(text)
        subtitles.append(parent_titles)
    return passages, subtitles        

# %%
doc_passages = {}
all_passages = []
start_idx = 0
for domain in docs['doc_data']:
    for doc_id in docs['doc_data'][domain].keys():
        ex = docs['doc_data'][domain][doc_id]
        #passages = split_text(ex["doc_text"]) # Token-based segmentation
        passages, subtitles = split_text_section(ex["spans"], ex["title"])
        all_passages.extend(passages)
        doc_passages[ex["doc_id"]] = (start_idx, len(passages))
        start_idx += len(passages)
        
passage_map = {}
for title in doc_passages:
    psg_start_ix = doc_passages[title][0]
    n_psgs = doc_passages[title][1]
    for i in range(n_psgs):
        passage_map[psg_start_ix + i] = {"text": all_passages[psg_start_ix + i], "title": title}

# %%
# encode documents
with open(os.path.join(_PATH,"retrieval_data/corpus.jsonl"), 'w') as fo:

    for passage_id, passage in passage_map.items():
        out_line = {k:v for k,v in passage.items()}
        out_line['_id'] = str(passage_id)
        fo.write(json.dumps(out_line) + "\n")

# %%
# encode queries
query_list = []

for split in ["train", "val", "test"]:
    print(split)
    queries_source_path = os.path.join(_PATH, "data", "mdd_all", "dd-generation-structure", f"{split}.source")
    qids_path = os.path.join(_PATH, "data", "mdd_all", "dd-generation-structure", f"{split}.qids")

    queries_data = open(queries_source_path, 'r')
    qids_data = open(qids_path, 'r')


    for query, qid in zip(queries_data, qids_data):
        out_line = {}
        # query = query.replace("||", " [SEP] ") # EXTRA SPACES DON'T MATTER
        out_line['text'] = query.strip()
        out_line['_id'] = qid.strip()
        query_list.append(out_line)

with open(os.path.join(_PATH,"retrieval_data/queries.jsonl"), 'w') as fo:
    for query in query_list:
        fo.write(json.dumps(query) + "\n")


# %%
# get qrels for validation set
os.makedirs(os.path.join(_PATH, "retrieval_data", "qrels"), exist_ok=True)
with open(os.path.join(_PATH, "retrieval_data", 'qrels', 'dev.tsv'), 'w') as fo:
    qids_path = os.path.join(_PATH, "data", "mdd_all", "dd-generation-structure/val.qids")
    pids_path = os.path.join(_PATH, "data", "mdd_all", "dd-generation-structure/val.pids")

    qids_data = open(qids_path, 'r')
    pids_data = open(pids_path, 'r')

    fo.write("query-id\tcorpus-id\tscore\n")

    for qid, pid in zip(qids_data, pids_data):
        qid = str(qid).strip()
        pid = str(pid).strip()
        fo.write('\t'.join([qid, pid, "1"]) + '\n')