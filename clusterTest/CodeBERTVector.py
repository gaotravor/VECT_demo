import os
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def java2vec():
    snippetList = []
    vectors = []
    lastVector = None
    className = open("csvFile/className.csv")
    for filename in className.readlines():
        file = open("javaFile" + "/" + filename.replace("\n", "").replace(".class", ".java"), errors='ignore')
        record = False
        lines = []
        for line in file.readlines():
            if "public static void main(String args[])" in line:
                record = True
            if record is True:
                lines.append(line.replace("\n", ""))
        lines = lines[2:-2]
        snippet = ""
        for line in lines:
            snippet += line.lstrip()
        snippetList.append(snippet)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    size = 0
    for snippet in snippetList:
        size = size + 1
        print(size)
        if len(snippet) == 0:
            vector = np.zeros(np.array(lastVector).size)
        elif len(snippet) >= 512:
            vector = np.ones(np.array(lastVector).size)
        else:
            nl_tokens = tokenizer.tokenize("")
            code_tokens = tokenizer.tokenize(snippet)
            tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            context_embeddings = model(torch.tensor(tokens_ids)[None, :])[0]
            vector = context_embeddings[0][0].detach().numpy()
            lastVector = vector
        vectors.append(vector)
    vectors = np.array(vectors)
    pd.DataFrame(vectors).to_csv("csvFile/CodeBERTVectors.csv")
    return vectors


vectors = java2vec()
