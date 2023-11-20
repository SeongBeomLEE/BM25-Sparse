import math
import numpy as np
import scipy.sparse as sp

class BaseBM25:
    def __init__(self, corpus:list[list[str]], verbose=False):
        if not all(isinstance(tokens, list) for tokens in corpus): raise Exception('토큰화된 Corpus가 아닙니다.')
        if not all(isinstance(token, str) for tokens in corpus for token in tokens): raise Exception('String이 아닌 token이 포함되어 있습니다.')
        self.corpus_size = len(corpus)
        self.verbose = verbose
        self._initialize(corpus)

    def _initialize(self, corpus:list[list[str]]):
        """
        TF, token_to_id 생성
        """
        if self.verbose:
            print("initialize Start")
        id = 0
        doc_len = []
        doc_freqs = []
        token_to_id = {}
        for tokens in corpus:
            frequencies = {}
            doc_len.append(len(tokens))
            for token in tokens:
                if token not in token_to_id:
                    token_to_id[token] = id
                    id += 1
                if token not in frequencies: 
                    frequencies[token] = 0
                frequencies[token] += 1
            doc_freqs.append(frequencies)

        self.token_to_id = token_to_id
        self.doc_freqs = doc_freqs
        self.avgdl = sum(doc_len) / len(corpus)
        self.doc_len = doc_len
        if self.verbose:
            print("initialize End")

    def _calc_TF(self) -> sp.csr_matrix:
        """
        TF 계산
        """
        if self.verbose:
            print("TF Calc Start")
        rows, cols, values = [], [], []
        row = 0
        for doc_freq in self.doc_freqs:
            for token in doc_freq.keys():
                rows.append(row)
                cols.append(self.token_to_id[token])
                values.append(doc_freq[token])
            row += 1
        TF = sp.csr_matrix((np.array(values), (np.array(rows), np.array(cols))), shape=(self.corpus_size, len(self.token_to_id)), dtype=np.float32)
        if self.verbose:
            print("TF Calc End")
        return TF
    
    def get_scores(self, tokens:list) -> np.array:
        if not all(isinstance(token, str) for token in tokens): raise Exception('String이 아닌 token이 포함되어 있습니다.')
        tokens = [self.token_to_id[token] for token in tokens if token in self.token_to_id]
        scores = np.squeeze(np.array(self.scores[:, tokens].sum(axis = 1)))
        return scores

    def get_most_similar(self, tokens:list, corpus:list[str], top:int=10) -> list[tuple[int, float]]:
        """
        가장 유사한 corpus의 id와 score을 반환 
        """
        kth = top
        if top >= self.corpus_size: 
            top = self.corpus_size
            kth = self.corpus_size - 1
        scores = self.get_scores(tokens)
        retrivals = (-scores).argpartition(kth=kth)[:top].tolist()
        return [corpus[retrival] for retrival in retrivals]

class BM25OkapiSparse(BaseBM25):
    """
    BM25Okapi의 Sparse Matrix 버전
    https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py
    """
    def __init__(self, corpus:list[list[str]], k1:float = 1.5, b:float = 0.75, epsilon:float = 0.25, verbose=False):
        super().__init__(corpus, verbose)
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.scores = self._calc_scores(self._calc_TF())
    
    def _calc_scores(self, TF:sp.csr_matrix) -> sp.csc_matrix:
        """
        Scores 계산
        """
        if self.verbose:
            print("Scores Calc Start")
        IDF = self._calc_IDF(TF)
        scores = TF.dot(sp.spdiags(np.array([self.k1 + 1] * len(IDF)), 0, len(IDF), len(IDF)))
        IDF = sp.spdiags(np.array(IDF), 0, len(IDF), len(IDF))
        scores = scores.dot(IDF)
        TF = self._calc_new_TF(TF)
        rows = scores.tocoo().row
        cols = scores.tocoo().col
        scores = np.squeeze(np.array(scores[scores.nonzero()] / TF[scores.nonzero()]))
        scores = sp.csr_matrix((scores, (rows, cols)), shape=(self.corpus_size, len(self.token_to_id)), dtype=np.float32).tocsc()
        if self.verbose:
            print("Scores Calc End")
        return scores

    def _calc_new_TF(self, TF:sp.csr_matrix) -> sp.csr_matrix:
        if self.verbose:
            print("New TF Calc Start")
        doc_len = self.k1 * (1 - self.b + self.b * np.array(self.doc_len) / self.avgdl)
        rows = TF.tocoo().row
        cols = TF.tocoo().col
        values = TF.tocoo().data
        new_TF = []
        for row, col, data in zip(rows, cols, values):
            new_TF.append(doc_len[row] + data)
        TF = sp.csr_matrix((np.array(new_TF), (rows, cols)), shape=(self.corpus_size, len(self.token_to_id)), dtype=np.float32)
        if self.verbose:
            print("New TF Calc End")
        return TF

    def _calc_IDF(self, TF:sp.csr_matrix) -> list:
        """
        IDF 계산
        """
        if self.verbose:
            print("IDF Calc Start")
        IDF = []
        negative_IDF_token_id_list = []
        frequencies = np.squeeze(np.array((TF >= 1).sum(0)))
        for token_id, freq in enumerate(frequencies):
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            IDF.append(idf)
            if idf < 0: negative_IDF_token_id_list.append(token_id)
        
        self.average_idf = sum(IDF) / len(self.token_to_id)
        eps = self.epsilon * self.average_idf
        for token_id in negative_IDF_token_id_list:
            IDF[token_id] = eps
        if self.verbose:
            print("IDF Calc End")
        return IDF
