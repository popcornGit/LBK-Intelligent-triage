# _*_ coding: UTF-8 _*_
# Author LBK
import pandas as pd
import numpy as np
from gensim import models
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import jieba
from gensim.models import LdaMulticore
from feature import label2idx
import gensim
import config


class SingletonMetaclass(type):
    """
    @description: singleton
    """
    def __init__(self, *args, **kwargs):
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.__instance is None:
            self.__instance = super(SingletonMetaclass, self).__call__(*args, **kwargs)

            return self.__instance
        else:
            return self.__instance


class Embedding(metaclass=SingletonMetaclass):
    def __init__(self):
        """
        @description: 这是Embedding类。可能多次被调用。我们需要使用单例模型。
        在这个类中，我们可以使用tfidf, word2vec, fasttext, autoencoder word embedding
        """

        # 停用词
        self.stopwords = [x.strip() for x in open('./data/stopwords.txt').readlines()]

    def load_data(self, path):
        """
        加载全部数据, 然后做分词
        :param path:
        :return:
        """
        data = pd.read_csv(path, sep='\t')
        data = data.fillna("")

        data["text"] = data['text'].apply(lambda )