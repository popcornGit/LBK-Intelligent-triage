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
            json.dump({k: v for k, v in labelToIndex.items()}, f)

    return labelToIndex

