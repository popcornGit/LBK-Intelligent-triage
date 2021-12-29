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

        data["text"] = data['text'].apply(lambda x: " ".join([w for w in x.split()
                                                              if w not in self.stopwords and w != '']))
        self.labelToIndex = label2idx(data)
        data["label"] = data['label'].map(self.labelToIndex)
        data["label"] = data.apply(lambda row: float(row['label']), axis=1)
        data = data[['text', 'label']]

        self.train = data['text'].tolist()

    def trainer(self):
        """
        @description: 训练 tfidf, word2vec, fasttext 喝 autoencoder
        :return:
        """
        count_vect = TfidfVectorizer(stop_words=self.stopwords,
                                     max_df=0.4,
                                     min_df=0.001,
                                     ngram_range=(1, 2))

        print(self.train[:5])
        self.tfidf = count_vect.fit(self.train)

        self.train = [sample.split() for sample in self.train]
        self.w2v = models.Word2Vec(min_count=2,
                                   window=5,
                                   vector_size=300,
                                   sample=6e-5,
                                   alpha=0.03,
                                   min_alpha=0.0007,
                                   negative=15,
                                   workers=4,
                                   epochs=30,
                                   max_vocab_size=50000)
        self.w2v.build_vocab(self.train)
        self.w2v.train(self.train,
                       total_examples=self.w2v.corpus_count,
                       epochs=15,
                       report_delay=1)

        self.id2word = gensim.corpora.Dictionary(self.train)
        corpus = [self.id2word.doc2bow(text) for text in self.train]
        self.LDAmodel = LdaMulticore(corpus=corpus,
                                     id2word=self.id2word,
                                     num_topics=30,
                                     workers=4,
                                     chunksize=4000,
                                     passes=7,
                                     alpha="asymmetric")

    def saver(self):
        """
        保存所有模型
        :return:
        """
        joblib.dump(self.tfidf, "./model/tfidf")

        self.w2v.wv.save_word2vec_format("./model/w2v.bin", binary=False)

        self.LDAmodel.save("./model/lda")

    def load(self):
        """
        加载所有Embedding模型
        :return:
        """
        self.tfidf = joblib.load("./model/tfidf")
        self.w2v = models.KeyedVectors.load_word2vec_format("./model/w2v.bin", binary=False)
        self.lda = models.ldamodel.LdaModel.load("./model/lda")


if __name__=="__main__":
    em = Embedding()
    em.load_data(config.train_data_file)
    em.trainer()
    em.saver()
