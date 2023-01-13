from bm25_sparse.tokenizer import Tokenizer
from bm25_sparse.bm25 import BM25OkapiSparse

corpus = [
    "Hello there good man!",
    "It is quite windy in London",
    "How is the weather today?"
]

tokenizer = Tokenizer()
tokenized_corpus = [tokenizer.tokenizer(doc) for doc in corpus]
bm25 = BM25OkapiSparse(tokenized_corpus)

query = "windy London"
tokenized_query = tokenizer.tokenizer(query)
scores = bm25.get_scores(tokenized_query)
print(scores)
most_similar_document = bm25.get_most_similar(tokenized_query, corpus, top=1)
print(most_similar_document)