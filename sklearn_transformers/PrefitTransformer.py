from sklearn.base import BaseEstimator, TransformerMixin

class PrefitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        return self.transformer.transform(X)