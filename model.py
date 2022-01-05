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
import pickle
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

    def feature_engineer(self, data, mode):
        if mode == "train":
            with open('data/basic_feature_train.pkl', 'rb') as f:
                data = pickle.load(f)
            return data

        if mode == "dev":
            with open("data/basic_feature_dev.pkl", 'rb') as f:
                data = pickle.load(f)
            return data

        data = get_tfidf(self.embedding.tfidf, data)
        data = get_embedding_feature(data, self.embedding.w2v)
        data = get_lda_features(data, self.embedding.lda)
        data = get_basic_feature(data)
        return data

    def trainer(self):
        self.train = self.feature_engineer(self.train, mode="train")
        self.dev = self.feature_engineer(self.dev, mode="dev")
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

        # print('X_train: ', X_train.shape, 'y_train: ', y_train.shape)
        # print(mlb.classes_)

        self.clf_BR = BinaryRelevance(classifier=lgb.LGBMClassifier(
                                        max_depth=5,
                                        learning_rate=0.1,
                                        n_estimators=100,
                                        silent=True,
                                        objective='binary',
                                        nthread=-1,
                                        reg_alpha=0,
                                        reg_lambda=1,
                                        # device='gpu',
                                        missing=None,
                                    ),
                                    require_dense=[False, True])

        self.clf_BR.fit(X_train, y_train)
        prediction = self.clf_BR.predict(X_test)
        print(prediction)
        print(y_test)
        print(metrics.accuracy_score(y_test, prediction))

    def save(self):
        joblib.dump(self.clf_BR, "./model/clf_BR")

    def load(self):
        self.model = joblib.load("./model/clf_BR")

    def predict(self, text):
        df = pd.DataFrame([[text]], columns=['text'])

        df["text"] = df["text"].apply(lambda x: " ".join(
            [w for w in jieba.cut(x) if w not in self.stopWords and w != ""]
        ))

        df = get_tfidf(self.embedding.tfidf, df)
        df = get_embedding_feature(df, self.embedding.w2v)
        df = get_lda_features(df, self.embedding.lda)
        df = get_basic_feature(df)

        cols = [x for x in df.columns if x not in self.exclusive_col]

        pred = self.model.predict(df[cols]).toarray()[0]

        result = [self.ix2label.get(i) for i in range(len(pred)) if pred[i] > 0]
        return result


if __name__ == "__main__":
    bc = Classifier(train_mode=True)
    # bc.trainer()
    # bc.save()
    bc.load()
    pred = bc.predict("张三有心脏病")
    print(pred)

