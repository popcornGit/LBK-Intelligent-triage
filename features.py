# _*_ coding: UTF-8 _*_
# Author LBK
import numpy as np
import pandas as pd
import joblib
import string
import jieba.posseg as pseg
import jieba
import json
import os


def label2idx(data):
    # 加载所有类别， 获取类别的embedding， 并保存文件
    if os.path.exists('./data/label2id.json'):
        labelToIndex = json.load(open('./data/label2id.json', encoding='utf-8'))

    else:
        label = data['label'].unique()
        labelToIndex = dict(zip(label, list(range(len(label)))))
        with open('./data/label2id.json', 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in labelToIndex.items()}, f, ensure_ascii=False)

    return labelToIndex


def get_tfidf(tfidf, data):
    stopWords = [x.strip() for x in open("./data/stopwords.txt").readlines()]
    text = data["text"].apply(lambda x: " ".join([w for w in x.split() if w not in stopWords and w != '']))
    data_tfidf = pd.DataFrame(
        tfidf.transform(text.tolist()).toarray()
    )
    data_tfidf.columns = ['tfidf' + str(i) for i in range(data_tfidf.shape[1])]
    data = pd.concat([data, data_tfidf], axis=1)
    # print(data.loc[0].values())

    return data


def array2df(data, col):
    return pd.DataFrame.from_records(
        data[col].values,
        columns=[col + "_" + str(i) for i in range(len(data[col].iloc[0]))]
    )


def get_embedding_feature(data, embedding_model):
    """
    word2vec -> max/mean, word2vec n-gram(2, 3, 4) -> max/mean, label embedding->max/mean
    :param data:
    :param embedding_model:
    :return:
    """
    labelToIndex = label2idx(data)

    w2v_label_embedding = np.array(
        [np.mean([embedding_model[word] for word in key
                  if word in embedding_model.index_to_key],
                 axis=0)
         for key in labelToIndex])

    joblib.dump(w2v_label_embedding, './data/w2v_label_embedding.pkl')

    tmp = data['text'].apply(lambda x: pd.Series(
        generate_feature(x, embedding_model, w2v_label_embedding)
    ))
    tmp = pd.concat([array2df(tmp, col) for col in tmp.colums], axis=1)
    data = pd.concat([data, tmp], axis=1)
    return data


def wam(sentence, w2v_model, method="mean", aggregate=True):
    """
    通过word average model 生成句子向量
    :param sentence: 以空格分隔的句子
    :param w2v_model: word2vec模型
    :param method: 聚合方法 mean 或者 max
    :param aggregate: 是否进行聚合
    :return:
    """
    arr = np.array([
        w2v_model[s] for s in sentence.strip().split()
        if s in w2v_model.index_to_key
    ])

    if not aggregate:
        return arr

    if len(arr) > 0:
        # 第一种方法对一条样本中的词求平均
        if method == "mean":
            return np.mean(np.array(arr), axis=0)
        # 第二种方法对一条样本中的词求最大
        elif method == "max":
            return np.max(np.array(arr), axis=0)
        else:
            raise NotImplementedError


def generate_feature(sentence, embedding_model, label_embedding):
    """
    word2vec -> max/mean, word2vec n-gram(2, 3, 4) -> max/mean, label embedding->max/mean
    :param sentence:
    :param embedding_model:
    :param label_embedding:
    :return:
    """
    # 首先在预训练的词向量中获取标签的词向量句子, 每一行表示一个标签表示
    # 每一行表示一个标签的embedding
    # 计算label embedding 具体参见文档

    # 同上, 获取embedding 特征, 不进行聚合
    w2v = wam(sentence, embedding_model, aggregate=False)  # [seq_len *300]

    print(len(w2v))
    if len(w2v) < 1:
        return {
            'w2v_label_mean': np.zeros(300),
            'w2v_label_max': np.zeros(300),
            'w2v_mean': np.zeros(300),
            'w2v_max': np.zeros(300),
            'w2v_2_mean': np.zeros(300),
            'w2v_3_mean': np.zeros(300),
            'w2v_4_mean': np.zeros(300),
            'w2v_2_max': np.zeros(300),
            'w2v_3_max': np.zeros(300),
            'w2v_4_max': np.zeros(300)
        }

    w2v_label_mean = Find_Label_embedding(w2v, label_embedding, method="mean")
    w2v_label_max = Find_Label_embedding(w2v, label_embedding, method="max")

    # 将embedding 进行 max, mean聚合
    w2v_mean = np.mean(np.array(w2v), axis=0)
    w2v_max = np.max(np.array(w2v), axis=0)

    # 滑窗处理embedding 然后聚合
    w2v_2_mean = Find_embedding_with_windows(w2v, 2, method="mean")
    w2v_3_mean = Find_embedding_with_windows(w2v, 3, method="mean")
    w2v_4_mean = Find_embedding_with_windows(w2v, 4, method="mean")

    w2v_2_max = Find_embedding_with_windows(w2v, 2, method="max")
    w2v_3_max = Find_embedding_with_windows(w2v, 3, method="max")
    w2v_4_max = Find_embedding_with_windows(w2v, 4, method="max")

    return {
            'w2v_label_mean': w2v_label_mean,
            'w2v_label_max': w2v_label_max,
            'w2v_mean': w2v_mean,
            'w2v_max': w2v_max,
            'w2v_2_mean': w2v_2_mean,
            'w2v_3_mean': w2v_3_mean,
            'w2v_4_mean': w2v_4_mean,
            'w2v_2_max': w2v_2_max,
            'w2v_3_max': w2v_3_max,
            'w2v_4_max': w2v_4_max
        }


def softmax(x):
    return np.exp(x) / np.exp(x).sum(axis=0)


def Find_Label_embedding(example_matrix, label_embedding, method="mean"):
    """
    获取标签空间的词嵌入
    :param example_matrix:
    :param label_embedding:
    :param method:
    :return:
    """
    # 根据矩阵乘法来计算label与word之间的相似度
    similarity_matrix = np.dot(example_matrix, label_embedding.T) / (
        np.linalg.norm(example_matrix) * np.linalg.norm(label_embedding)
    )

    # 然后对相似矩阵进行均值池化，则得到了“类别-词语”的注意力机制
    # 这里可以使用max-pooling和mean-pooling
    attention = similarity_matrix.max(axis=1)
    attention = softmax(attention).reshape(-1, 1)
    # 将样本的词嵌入与注意力机制相乘得到
    attention_embedding = example_matrix * attention

    if method == 'mean':
        return np.mean(attention_embedding, axis=0)
    else:
        return np.max(attention_embedding, axis=0)



def Find_embedding_with_windows(example_matrix, window_size=2, method="mean"):
    return 1