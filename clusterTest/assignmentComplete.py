import os
import sys

import pandas as pd
import numpy as np

FILE_SEPARATOR = os.path.sep

modelName = sys.argv[1]

result = pd.read_csv("clusterFile" + FILE_SEPARATOR + modelName + "Result.csv")
result = result.drop(labels='Unnamed: 0', axis=1)  # 删除没啥用的第一列
result = result.values
maxCluster = np.max(result)

df = pd.read_csv("csvFile" + FILE_SEPARATOR + modelName + "Vectors.csv")

df = df.drop(labels='Unnamed: 0', axis=1)  # 删除没啥用的第一列

featureList = df.values
featureList = np.array(featureList)
index = 0
for i in featureList:
    if (np.array(i) == 1).all():
        result = np.insert(result, index, maxCluster + 1)
    if (np.array(i) == 0).all():
        result = np.insert(result, index, maxCluster + 2)
    index = index + 1
pd.DataFrame(result).to_csv("csvFile" + FILE_SEPARATOR + modelName + "Assignment.csv", index=False, header=False)
