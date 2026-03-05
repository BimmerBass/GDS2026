from dataclasses import dataclass, field
from nltk.stem.api import StemmerI
from collections import Counter
from typing import Any
from tqdm import tqdm
import pandas as pd
import numpy as np
import re

@dataclass
class TokenizerConfig:
    top_n: int = 0
    special_tokens: list[str] = field(default_factory=lambda: [])
    tokens: np.ndarray = field(default_factory=lambda: [])
    token_frequencies: np.ndarray = field(default_factory=lambda: [])
    tokens_to_ids: dict[str,int] = field(default_factory=lambda: {})
    stopwords: list[str] = field(default_factory=lambda: [])
    stemmer: StemmerI | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_n": self.top_n,
            "special_tokens": self.special_tokens,
            "tokens_to_ids": self.tokens_to_ids,
            "stopwords": self.stopwords,
            "stemmer": type(self.stemmer).__name__,
            "tokens": self.tokens.tolist(),
            "token_frequencies": self.token_frequencies.tolist()
        }


class Tokenizer:
    def __init__(self, config : TokenizerConfig, mutable: bool):
        self.__config = config
        self.__mutable = mutable
    @classmethod
    def from_config(cls, config : TokenizerConfig):
        return cls(config, False)
    def get_config(self) -> TokenizerConfig:
        return self.__config
    @classmethod
    def create_and_fit(
        cls, 
        column: pd.Series,
          top_n : int, 
          special_tokens : list[str],
          stopwords : list[str],
          stemmer : StemmerI):
        instance = cls(TokenizerConfig(top_n, special_tokens, stopwords=stopwords, stemmer=stemmer), True)
        instance.fit_tokenizer(column)
        return instance

    def fit_tokenizer(self, column: pd.Series) -> int:
        if not self.__mutable:
            raise RuntimeError("cannot fit an immutable tokenizer")
        
        counter = Counter()
        for text in tqdm(column, total=len(column)):
            tokens = self.__tokenize(text)
            counter.update(tokens)

        most_common_n = counter.most_common(self.__config.top_n)
        self.__config.tokens = np.array([tk for tk, _ in most_common_n])
        self.__config.token_frequencies = np.array([i for _, i in most_common_n])
        self.__config.tokens_to_ids = {inx : tk.item() for inx,tk in enumerate(self.__config.tokens)}
        self.__mutable = False
        return len(most_common_n)

    def text_to_ids(self, text : str) -> np.ndarray:
        terms = self.__tokenize(text)
        tokens = [self.__config.tokens_to_ids[term] for term in terms if term in self.__config.tokens_to_ids]
        return np.array(tokens)
    
    def size(self) -> int:
        return self.__config.tokens.size
    
    def get_id_from_token(self, token : str) -> int | None:
        if not token in self.__config.tokens_to_ids:
            return None
        return self.__config.tokens_to_ids[token]
    
    def get_token_from_id(self, id : int) -> str | None:
        if id >= self.__config.tokens.size:
            return None
        return self.__config.tokens[id]

    def __tokenize(self, text : str) -> list[str]:
        text = str(text)
        no_punct = re.sub(r"[^\w<> ]", '', text)    # remove all characters that aren't an alphanumerical one, a space or < / >
        no_ws = re.sub(r"\s+", " ", no_punct) # collapse whitespaces to single ' '
        terms = no_ws.split(" ") # split into terms -> candidate tokens

        trimmed = [term.strip() for term in terms]
        tokens = [term for term in trimmed if term != "" and term not in self.__config.stopwords]
        if self.__config.stemmer != None:
            tokens = [self.__config.stemmer.stem(token) for token in tokens]
        return tokens