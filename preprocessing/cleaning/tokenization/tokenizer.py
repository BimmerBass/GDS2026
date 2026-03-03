from dataclasses import dataclass
import pandas as pd
import numpy as np
import re

@dataclass
class TokenizerConfig:
    top_n: int = 0
    special_tokens: list[str] = []
    tokens: np.ndarray = []
    token_frequencies: np.ndarray = []
    tokens_to_ids: dict[str,int] = {}
    stopwords: list[str] = []

class Tokenizer:
    def __init__(self, config : TokenizerConfig, mutable: bool):
        self.__config = config
        self.__mutable = mutable
    @classmethod
    def from_config(cls, config : TokenizerConfig):
        return cls(config, False)
    @classmethod
    def create(cls, top_n : int, special_tokens : list[str]):
        return cls(TokenizerConfig(top_n, special_tokens), True)

    def fit_tokenizer(self, column: pd.Series) -> int:
        if not self.__mutable:
            raise RuntimeError("cannot fit an immutable tokenizer")
        


        self.__mutable = False

    def __tokenize(self, text : str) -> list[str]:
        no_punct = re.sub(r"[^\w<> ]", '', text)
        no_ws = re.sub(r"\s+", " ", text)

        terms = no_ws.split(" ")