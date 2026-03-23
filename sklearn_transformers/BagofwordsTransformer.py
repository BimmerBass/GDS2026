from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np

class BagofwordsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column : str, vocab_size: int):
        super().__init__()
        self.column = column
        self.vocab_size = vocab_size

    def fit(self, X,y=None):
        return self
    
    def transform(self, X : pd.DataFrame) -> np.ndarray:
        rows,cols, data = [], [], []

        docs = X[self.column]
        for i,tokens in tqdm(enumerate(docs), total=len(docs)):
            if len(tokens) < 1:
                continue
            counts = np.bincount(tokens, minlength=self.vocab_size)
            nz = np.nonzero(counts)[0]
            rows.extend([i]*len(nz))
            cols.extend(nz.tolist())
            data.extend(counts[nz].tolist())
        return csr_matrix((data, (rows, cols)), shape=(len(docs), self.vocab_size), dtype=np.int32)
