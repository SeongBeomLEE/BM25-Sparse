import os
import json
import pandas as pd
import logging

import math
import numpy as np
import scipy.sparse as sp
from datetime import datetime

class BM25:
    """
    https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py
    https://velog.io/@mayhan/Elasticsearch-%EC%9C%A0%EC%82%AC%EB%8F%84-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98
    https://stackoverflow.com/questions/15755270/scipy-sparse-matrices-purpose-and-usage-of-different-implementations
    """
    def __init__(self, corpus):
        self.corpus_size = 0
        self.word_to_index = {}
        self.k1 = 1.5
        self.b = 0.75
        self.epsilon = 0.25
        self.make_word_to_index(corpus)
        self._calc_TF_IDF(corpus)
        
    def make_word_to_index(self, corpus):
        index = 0
        num_doc = 0
        doc_len = []
        
        print(f"Make word_to_index Start: {datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
        start = datetime.now()
        
        for document in corpus:
            doc_len.append(len(document))
            num_doc += len(document)
            for word in document:
                if word not in self.word_to_index:
                    self.word_to_index[word] = index
                    index += 1
            
            self.corpus_size += 1
            
        end = datetime.now()
        print(f"Make word_to_index End: {end - start}")
        
        self.avgdl = num_doc / self.corpus_size
        self.doc_len = np.array(doc_len)

    def _calc_TF_IDF(self, corpus):
        
        print(f"TF Cal Start: {datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
        start = datetime.now()
        
        datas = {}
        for corpus_index, document in enumerate(corpus):
            for word in document:
                if (corpus_index, self.word_to_index[word]) not in datas:
                    datas[(corpus_index, self.word_to_index[word])] = 1
                else:
                    datas[(corpus_index, self.word_to_index[word])] += 1
        
        rows = []
        cols = []
        values = []
        for key, value in datas.items():
            rows.append(key[0])
            cols.append(key[1])
            values.append(value)
        
        TF = sp.csr_matrix((np.array(values), (np.array(rows), np.array(cols))), shape=(self.corpus_size, len(self.word_to_index)), dtype=np.float32)

        del rows, cols, values
        
        end = datetime.now()
        print(f"TF Cal End: {end - start}")
        
        print(f"IDF Cal Start: {datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
        start = datetime.now()
        
        frequencies = np.squeeze(np.array((TF >= 1).sum(0)))
        IDF = []
        negative_IDF_index_list = []
        idf_sum = 0
        for word_index, freq in enumerate(frequencies):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            IDF.append(idf)
            idf_sum += idf
            if idf < 0:
                negative_IDF_index_list.append(word_index)
        
        self.average_idf = idf_sum / len(self.word_to_index)
        eps = self.epsilon * self.average_idf
        for word_index in negative_IDF_index_list:
            IDF[word_index] = eps
            
        end = datetime.now()
        print(f"IDF Cal End: {end - start}")
        
        print(f"Score Cal Start: {datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
        start = datetime.now()

        score = TF.dot(sp.spdiags(np.array([self.k1 + 1] * len(IDF)), 0, len(IDF), len(IDF)))
        IDF = sp.spdiags(np.array(IDF), 0, len(IDF), len(IDF))
        score = score.dot(IDF)
        
        doc_len = self.doc_len
        doc_len = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
        TF_rows = TF.tocoo().row
        TF_cols = TF.tocoo().col
        TF_datas = TF.tocoo().data
        TF_new_datas = []
        
        for row, col, data in zip(TF_rows, TF_cols, TF_datas):
            TF_new_datas.append(doc_len[row] + data)
        
        TF = sp.csr_matrix((np.array(TF_new_datas), (TF_rows, TF_cols)), shape=(self.corpus_size, len(self.word_to_index)), dtype=np.float32)
        
        score_rows = score.tocoo().row
        score_cols = score.tocoo().col
        score = np.squeeze(np.array(score[score.nonzero()] / TF[score.nonzero()]))
        self.score = sp.csr_matrix((score, (score_rows, score_cols)), shape=(self.corpus_size, len(self.word_to_index)), dtype=np.float32).tocsc()
        end = datetime.now()
        print(f"Score Cal End: {end - start}")
        
    def get_scores(self, query):
        query_to_index = [self.word_to_index[q] for q in query]
        score = self.score[:, query_to_index]
        return score.sum(axis = 1)

def load_data(data_load_path:str, data_name:str):
    data_path = os.path.join(data_load_path, data_name)
    with open(data_path, "r") as json_data:
        data = json.load(json_data)["data"]
    print(f"{data_name} Data Load {len(data)}")
    return pd.DataFrame(data)

def save_data(data:dict, data_save_path:str, data_name:str):
    data_path = os.path.join(data_save_path, data_name + ".json")
    with open(data_path, 'w') as outfile:
        json.dump(data, outfile, ensure_ascii=False)
    print(f"{data_name} Data Save {len(data)}")


def main():
    # 로그 생성
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler('my.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
        
    data_path = "/data2/sblee/bert-serch/train-data"
    logger.info(f'Data Load Start')
    train_context_df = load_data(data_path, "train-context.json")
    logger.info(f'Data Load End')
    
    logger.info(f'corpus Load Start')
    corpus_index_to_item_context_id = []
    corpus = []
    for index, df in train_context_df.iterrows():
        item_context_id = df["item-context_id"]
        clean_tokens = df["clean-tokens"]
        
        corpus_index_to_item_context_id.append(item_context_id)
        corpus.append(clean_tokens)
    
    save_data({"data" : corpus_index_to_item_context_id}, data_path, "corpus_index_to_item_context_id")
    logger.info(f'corpus Load End')
    
    logger.info(f'BM25 Load Start')
    bm25 = BM25(corpus)
    logger.info(f'BM25 Load End')
    
    total_count = len(corpus)
    count = 0
    corpus_scores = []
    for c in corpus:
        score = -np.squeeze(np.array(bm25.get_scores(c)))
        score = score.argpartition(kth=11)[:11].tolist()
        corpus_scores.append(score)
        count += 1
        
        if count % 10000 == 0 or count == total_count:
            logger.info(f'{count} / {total_count}')
    
    save_data({"data" : corpus_scores}, data_path, "corpus_scores")

if __name__ == '__main__':
    main()