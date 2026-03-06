from dataclasses import dataclass, field
from cleantext import clean
import polars as pl
import pandas as pd
from tqdm import tqdm
import numpy as np
import re

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
    text = str(text)

    if config.lower:
        text = text.lower()
    text = re.sub(r"[<>]", " ", text)

    return clean(
        text,
        lower=False, # do not lowercase now, since masked tokens will then also be lowercased.
        no_line_breaks=config.collapse_whitespace,
        no_urls=True,
        no_emails=True,
        no_numbers=True,
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_number="<NUM>")

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