from dataclasses import dataclass, field
import polars as pl
import pandas as pd

@dataclass
class CleaningConfig:
    lower: bool = True
    collapse_whitespace: bool = True
    regexes: dict[str, str] = field(default_factory=lambda: {
        r'[a-zA-Z]+.?\s*\d{1,2}[,.]?\s*\d{4}': "<DATE>",
        r'[\w.]+@[\w.]+': "<EMAIL>",
        r'([^\W\d]*)://(((\w*@)?([\w-]+(\.[\w-]+)+)(:[\d]+)?))?(/[\w-]+)*\/?(?:\?[^\s#]*)?(#[\w-]+)?': "<URL>",
        r'\d[\d.,]*': "<NUM>"
    })


def clean_text(text : str, config: CleaningConfig) -> str:
    df = pd.DataFrame({
        "content": [text],
        "title": [""]
    })
    cleaned = clean_data(df, config)
    return cleaned.loc[0,"content"]

def clean_data(df: pd.DataFrame, config: CleaningConfig) -> pd.DataFrame:
    pdf = pl.from_pandas(df)
    expr = pl.col(["title", "content"])

    if config.lower:
        expr = expr.str.to_lowercase()

    expr = expr.str.replace_all(r"[<>]", " ")

    for regex, repl in config.regexes.items():
        expr = expr.str.replace_all(regex, repl)
    expr = expr.str.replace_all(r"\s+", " ") # collapse whitespace

    pdf = pdf.with_columns(expr)
    return pdf.to_pandas(use_pyarrow_extension_array=True)

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