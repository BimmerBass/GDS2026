from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from typing import Sequence
from tqdm import tqdm
import pandas as pd
import numpy as np

class BagofwordsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def fit(self, X,y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X : Sequence[list[int]]) -> np.ndarray:
        rows,cols, data = [], [], []

        for i,tokens in tqdm(enumerate(X), total=len(X)):
            if len(tokens) < 1:
                continue
            counts = np.bincount(tokens, minlength=self.vocab_size)
            nz = np.nonzero(counts)[0]
            rows.extend([i]*len(nz))
            cols.extend(nz.tolist())
            data.extend(counts[nz].tolist())
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocab_size), dtype=np.int32)
