from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np

class CatColPrependTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column : str):
        super().__init__()
        self.column = column

    def fit(self, X : pd.DataFrame, y = None):
        self.is_fitted_ = True
        return self