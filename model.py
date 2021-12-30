# _*_ coding: UTF-8 _*_
# Author LBK
import json
import jieba
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance

from embedding import Embedding
from features import (get_tfidf, get_embedding_feature)


class Classifier:
    def __init__(self, train_mode=False) -> None:
        self.stopWords = [x.strip() for x in open('./data/stopwords.txt').readlines()]

        self.embedding = Embedding()
        self.embedding.load()
        self.labelToIndex = json.load(open('./data/label2id.json', encoding='utf-8'))
        self.ix2label = {v: k for k, v in self.labelToIndex.items()}

        if train_mode:
            self.train = pd.read_csv('./data/data_cat10_annotated_train.txt',
                                     sep='\t',
                                     header=None,
                                     names=["label", "text"]).dropna().reset_index(drop=True)
            self.dev = pd.read_csv('./data/data_cat10_annotated_eval.txt',
                                   sep='\t',
                                   header=None,
                                   names=["label", "text"]).dropna().reset_index(drop=True)
            self.test = pd.read_csv('./data/data_cat10_annotated_test.txt',
                                    sep='\t',
                                    header=None,
                                    names=["label", "text"]).dropna().reset_index(drop=True)

            te_shu = json.load(open('./data/te_shu.json', encoding='utf-8'))
            self.train["label"] = self.train.apply(lambda row: str(row['label']), axis=1)
            self.dev["label"] = self.dev.apply(lambda row: str(row['label']), axis=1)
            self.test["label"] = self.test.apply(lambda row: str(row['label']), axis=1)
            self.train["label"] = self.train['label'].map(te_shu)
            self.dev["label"] = self.dev['label'].map(te_shu)
            self.test["label"] = self.test['label'].map(te_shu)

        self.exclusive_col = ["text", "lda", "bow", "label"]

    def feature_engineer(self, data):
        data = get_tfidf(self.embedding.tfidf, data)
        # print(self.embedding.w2v.index_to_key)
        data = get_embedding_feature(data, self.embedding.w2v)

        return data

    def trainer(self):
        self.train = self.feature_engineer(self.train)


if __name__ == "__main__":
    bc = Classifier(train_mode=True)
    bc.trainer()
    # bc.save()