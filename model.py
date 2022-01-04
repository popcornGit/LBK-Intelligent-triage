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
from features import (get_tfidf, get_embedding_feature,
                      get_lda_features, get_basic_feature)


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
        data = get_embedding_feature(data, self.embedding.w2v)
        data = get_lda_features(data, self.embedding.lda)
        data = get_basic_feature(data)
        return data

    def trainer(self):
        self.train = self.feature_engineer(self.train)
        self.dev = self.feature_engineer(self.dev)
        cols = [x for x in self.train.columns if x not in self.exclusive_col]

        X_train = self.train[cols]
        y_train = self.train["label"]

        X_test = self.dev[cols]
        y_test = self.dev["label"]

        mlb = MultiLabelBinarizer(sparse_output=False)

        y_train_new = [[i] for i in y_train]
        y_test_new = [[i] for i in y_test]

        y_train = mlb.fit_transform(y_train_new)
        y_test = mlb.transform(y_test_new)

        print('X_train: ', X_train.shape, 'y_train: ', y_train.shape)
        print(mlb.classes_)


if __name__ == "__main__":
    bc = Classifier(train_mode=True)
    bc.trainer()
    # bc.save()