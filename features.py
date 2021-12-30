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
        [np.mean([embedding_model.wv.get_vector(word) for word in key
                  if word in embedding_model.wv.vocab.keys()],
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
        w2v_model.wv.get_vector(s) for s in sentence
        if s in w2v_model.wv.vocab.keys()
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

    else:
        return np.zeros(300)


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
