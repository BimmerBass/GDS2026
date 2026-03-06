from .clean_data import *
import pandas as pd

def filter_dataset(
        df : pd.DataFrame,
        drop_cols: list[str],
        remove_nulls_cols: list[str],
        deduplicate_cols: list[str],
        convert_to_cat_cols: list[str]) -> pd.DataFrame:
    dropped = df.drop(columns=drop_cols)
    for col in deduplicate_cols:
        dropped[col] = dropped[col].drop_duplicates()
    for col in convert_to_cat_cols:
        dropped[col] = dropped[col].astype("category")

    dropped = dropped.dropna(axis=0, subset=remove_nulls_cols)
    return dropped
