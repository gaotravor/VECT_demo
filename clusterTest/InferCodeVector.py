from infercode.client.infercode_client import InferCodeClient
import logging
import numpy as np
import pandas as pd
import os

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
    inferCode = InferCodeClient(language="java")
    inferCode.init_from_config()
    size = 0
    for snippet in snippetList:
        size = size + 1
        print(size)
        if len(snippet) == 0:
            vector = np.zeros(np.array(lastVector).size)
        elif len(snippet) >= 512:
            vector = np.ones(np.array(lastVector).size)
        else:
            vector = inferCode.encode([snippet])
            vector = vector[0]
            lastVector = vector
        vectors.append(vector)
    vectors = np.array(vectors)
    pd.DataFrame(vectors).to_csv("csvFile/InferCodeVectors.csv")
    print(vectors)
    return vectors


vectors = java2vec()
