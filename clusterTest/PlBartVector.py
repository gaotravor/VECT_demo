import os
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import numpy as np
import logging
import pandas as pd
from transformers import PLBartConfig, PLBartTokenizer, PLBartModel


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x, **kwargs):
        x = x.reshape(-1, x.size(-1) * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class CloneModel(nn.Module):
    def __init__(self, encoder, config, tokenizer):
        super(CloneModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.eos_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.bos_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        if self.args.model_type == 'codet5':
            vec = self.get_t5_vec(source_ids)
        elif self.args.model_type == 'bart':
            vec = self.get_bart_vec(source_ids)
        elif self.args.model_type == 'roberta':
            vec = self.get_roberta_vec(source_ids)

        logits = self.classifier(vec)
        prob = nn.functional.softmax(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob


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
    size = 0
    config = PLBartConfig.from_pretrained("uclanlp/plbart-base")
    tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
    encoder = PLBartModel.from_pretrained("uclanlp/plbart-base")
    for snippet in snippetList:
        size = size + 1
        print(size)
        if len(snippet) == 0:
            vector = np.zeros(np.array(lastVector).size)
        elif len(snippet) >= 512:
            vector = np.ones(np.array(lastVector).size)
        else:
            model = CloneModel(encoder, config, tokenizer)
            input_ids = tokenizer.encode(snippet, return_tensors="pt")
            vector = model.get_bart_vec(input_ids)
            vector = vector[0].detach().numpy()
            lastVector = vector
        vectors.append(vector)
    vectors = np.array(vectors)
    pd.DataFrame(vectors).to_csv("csvFile/PlBartVectors.csv")
    return vectors


logging.basicConfig(level=logging.INFO)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

vectors = java2vec()
