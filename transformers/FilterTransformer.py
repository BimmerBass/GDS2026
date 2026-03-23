from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any
import pandas as pd

class FilterTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            drop_cols : list[str],
            remove_nulls_col_names : list[str],
            deduplicate_cols : list[str],
            convert_to_category_cols: list[str],
            remove_cols_with_value: dict[str, Any]):
        super().__init__()
        self.drop_cols = drop_cols
        self.remove_nulls_col_names = remove_nulls_col_names
        self.deduplicate_cols = deduplicate_cols
        self.convert_to_category_cols = convert_to_category_cols
        self.remove_cols_with_value = remove_cols_with_value

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        dropped = X.drop(columns=self.drop_cols, errors="ignore")
        dropped = dropped.drop_duplicates(subset=self.deduplicate_cols)
        for column in self.convert_to_category_cols:
            dropped[column] = dropped[column].astype("category")
        for column in self.remove_cols_with_value:
            val = self.remove_cols_with_value[column]
            dropped = dropped[dropped[column] != val]
        dropped = dropped.dropna(axis=0, subset=self.remove_nulls_col_names)
        return dropped[pd.to_numeric(dropped["id"], errors="coerce").notna()]