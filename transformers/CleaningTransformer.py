from sklearn.base import BaseEstimator, TransformerMixin
import polars as pl
import pandas as pd

class CleaningTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            lowercase: bool = True,
            collapse_whitespace: bool = True,
            replace_patterns : dict[str,str] = {
                r'[a-zA-Z]+.?\s*\d{1,2}[,.]?\s*\d{4}': "<DATE>",
                r'[\w.]+@[\w.]+': "<EMAIL>",
                r'([^\W\d]*)://(((\w*@)?([\w-]+(\.[\w-]+)+)(:[\d]+)?))?(/[\w-]+)*\/?(?:\?[^\s#]*)?(#[\w-]+)?': "<URL>",
                r'\d[\d.,]*': "<NUM>" }):
        super().__init__()
        self.lowercase = lowercase
        self.collapse_whitespace = collapse_whitespace
        self.replace_patterns = replace_patterns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        pdf = pl.from_pandas(X)
        expr = pl.col(["title", "content"])

        if self.lowercase:
            expr = expr.str.to_lowercase()
        expr = expr.str.replace_all(r"[<>]", " ")

        for regex, repl in self.replace_patterns.items():
            expr = expr.str.replace_all(regex, repl)
        if self.collapse_whitespace:
            expr = expr.str.replace_all(r"\s+", " ")

        pdf = pdf.with_columns(expr)
        return pdf.to_pandas(use_pyarrow_extension_array=True)