import itertools
from collections import Counter

import numpy as np


def safeLog(x):
    mask = x != 0
    x[mask] = np.log(x[mask])
    return x

class TF_IDF():
    # tf计算加权方法
    tf_methods = {
        "log": lambda x: np.log(1+x),
        "augmented": lambda x: 0.5 + 0.5 * x / np.max(x, axis=1, keepdims=True),
        "boolean": lambda x: np.minimum(x, 1),
        "log_avg": lambda x: (1 + safeLog(x)) / (1 + safeLog(np.mean(x, axis=1, keepdims=True))),
    }

    def __init__(self, corpus:list):
        self.corpus = corpus
        # 去掉“,”，并把语料库拆成单词
        self.corpus_words = [doc.replace(",", "").split(" ") for doc in corpus]
        # 词汇表
        self.vocab = set(itertools.chain(*self.corpus_words))
        # 索引词汇表便于画图
        self.v2i = {v: i for i, v in enumerate(self.vocab)}
        self.i2v = {i: v for v, i in self.v2i.items()}

    def computeTF(self):
        # 初始化空矩阵：[n_vocab, n_doc]
        tf = np.zeros((len(self.vocab), len(self.corpus)), dtype=np.float64)
        # 遍历corpus
        for i, d in enumerate(self.corpus_words):
            # 统计每个文档中单词出现次数
            counter = Counter(d)
            # 计算文档总词数
            doc_words_cnt = float(len(d))
            # 遍历文档中每个词
            for v in counter.keys():
                # 计算该词的tf值
                tf[self.v2i[v], i] = counter[v] / doc_words_cnt
        return tf

    def computeIDF(self):
        # 初始化矩阵：[n_vocab, 1]
        idf = np.zeros((len(self.vocab), 1))
        # 遍历词汇表中的每个单词
        for i in range(len(self.vocab)):
            d_cnt = 0 # 该词在文档中出现的次数
            # 遍历语料库中的每个文档
            for d in self.corpus_words:
                # 计算该词在文档中出现的次数
                d_cnt += 1 if self.i2v[i] in d else 0
            # 计算该词的IDF
            idf[i, 0] = np.log(len(self.corpus)/(d_cnt + 1))
        return  idf

    def computeTFIDF(self, tf, idf):
        return tf * idf


def main():
    corpus = [
        "it is a good day, I like to stay here",
        "I am happy to be here",
        "I am bob",
        "it is sunny today",
        "I have a party today",
        "it is a dog and that is a cat",
        "there are dog and cat on the tree",
        "I study hard this morning",
        "today is a good day",
        "tomorrow will be a good day",
        "I like coffee, I like book and I like apple",
        "I do not like it",
        "I am kitty, I like bob",
        "I do not care who like bob, but I like kitty",
        "It is coffee time, bring your cup",
    ]
    tfidf = TF_IDF(corpus)
    tf_score = tfidf.computeTF()
    idf_score = tfidf.computeIDF()
    tfidf_score = tfidf.computeTFIDF(tf_score, idf_score)
    for d in tfidf_score:
        print(d)


if __name__=="__main__":
    main()
