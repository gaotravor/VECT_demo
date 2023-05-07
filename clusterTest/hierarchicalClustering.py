import os
import sys
import pandas
import pandas as pd
import sklearn.cluster as sc
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from tqdm import tqdm


def getLinkageMat(model):
    children = model.children_
    cs = np.zeros(len(children))
    N = len(model.labels_)
    for i, child in enumerate(children):
        count = 0
        for idx in child:
            count += 1 if idx < N else cs[idx - N]
        cs[i] = count
    return np.column_stack([children, model.distances_, cs])


FILE_SEPARATOR = os.path.sep

modelName = sys.argv[1]

df = pd.read_csv("csvFile" + FILE_SEPARATOR + modelName + "Vectors.csv")

df = df.drop(labels='Unnamed: 0', axis=1)  # 删除没啥用的第一列

featureList = df.values
featureList = np.array(featureList)
index = 0
skipIndex = []
isCluster = []
for i in featureList:
    if (np.array(i) == 1).all():
        skipIndex.append(index)
        isCluster.append(1)
    if (np.array(i) == 0).all():
        skipIndex.append(index)
        isCluster.append(0)
    index = index + 1
featureList = np.delete(featureList, skipIndex, 0)
featureList = normalize(featureList)

# 聚类
model = sc.AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(featureList)
maxScore = -1
maxCluster = -1
maxResult = []
# 构造树状图
mat = getLinkageMat(model)
scoreList = []
for i in tqdm(range(1, len(mat) - 1, 100)):
    # 根据树状图求解每层聚类的结果
    result = fcluster(mat, t=i + 1, criterion="maxclust")
    score = metrics.silhouette_score(featureList, result)
    scoreList.append(score)
    if score > maxScore:
        maxScore = score
        maxCluster = i + 1
        maxResult = result
    print()
    print(maxScore)
    print(maxCluster)
    print(maxResult)
    print("_______________________________")
    csvFile = np.array([maxResult])
    df = pandas.DataFrame(csvFile.transpose())
    df.to_csv("clusterFile" + FILE_SEPARATOR + modelName + "Result.csv")
    plt.plot(scoreList)
    plt.savefig("clusterFile" + FILE_SEPARATOR + modelName + "Score.jpg")


