# BM25-Sparse
- 기존의 [rank_bm25](https://github.com/dorianbrown/rank_bm25) 대용량 corpus에 대하여 Out of Memory 문제가 발생해, 이를 해결하고자 numpy로 구현된 기존의 rank_bm25를 scipy 형태로 구현했습니다.
- 추가적으로 rank_bm25와 달리 전체 corpus에 대하여 score를 미리 계산하여, most similar document의 탐색 속도를 개선했습니다.

## Installation

```bash
git clone https://github.com/SeongBeomLEE/BM25-Sparse.git
```

## Usage

### Initalizing
```python
from bm25_sparse.tokenizer import Tokenizer
from bm25_sparse.bm25 import BM25OkapiSparse

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenizer = Tokenizer()
tokenized_corpus = [tokenizer.tokenize(doc) for doc in corpus]
bm25 = BM25OkapiSparse(tokenized_corpus)
```

### Ranking of documents
```python
query = "windy London"
tokenized_query = tokenizer.tokenize(query)
scores = bm25.get_scores(tokenized_query)
# array([0.        0.9372948 0.       ])
```

```python
query = "windy London"
tokenized_query = tokenizer.tokenize(query)
most_similar_document = bm25.get_most_similar(tokenized_query, corpus, top=1)
['It is quite windy in London']
```

## TODO
- [x] Okapi BM25
- [ ] pip install
- [ ] 설명
- [ ] 병렬 처리
- [ ] BM25L
- [ ] BM25+
- [ ] BM25-Adpt
- [ ] BM25T 
