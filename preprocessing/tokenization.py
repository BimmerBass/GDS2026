from dataclasses import dataclass, field
from typing import Any, Iterable
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import Stemmer
import json
import re

@dataclass
class TokenizerConfig:
    top_n: int = 0
    special_tokens: list[str] = field(default_factory=lambda: [])
    tokens: np.ndarray = field(default_factory=lambda: [])
    token_frequencies: np.ndarray = field(default_factory=lambda: [])
    tokens_to_ids: dict[str,int] = field(default_factory=lambda: {})
    stopwords: set[str] = field(default_factory=lambda: set())
    stemmer: Stemmer.Stemmer = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "top_n": self.top_n,
            "special_tokens": self.special_tokens,
            "tokens": self.tokens.tolist(),
            "token_frequencies": self.token_frequencies.tolist(),
            "tokens_to_ids": self.tokens_to_ids,
            "stopwords": list(self.stopwords),
            "stemmer": "pystemmer" if self.stemmer is not None else None
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)
    
    @classmethod
    def from_json(cls, jsons: str):
        data = json.loads(jsons)

        data["tokens"] = np.array(data["tokens"])
        data["token_frequencies"] = np.array(data["token_frequencies"])
        data["stopwords"] = set(data["stopwords"])
        data["stemmer"] = Stemmer.Stemmer("english") if data["stemmer"] is not None else None

        return cls(**data)


class Tokenizer:
    def __init__(self, config : TokenizerConfig, mutable: bool):
        self.__config = config
        self.__mutable = mutable

    @classmethod
    def from_json(cls, filepath: str, mutable: bool):
        with open(filepath, "r", encoding="utf-8") as file:
            filedata = file.read()
        return cls(TokenizerConfig.from_json(filedata), mutable)
    def to_json(self, outputfile: str) -> None:
        with open(outputfile, "w", encoding="utf-8") as file:
            file.write(self.__config.to_json())


    @classmethod
    def create_and_fit(
        cls, 
        column: pd.Series,
          top_n : int, 
          special_tokens : list[str],
          stopwords : set[str],
          stemmer : Stemmer.Stemmer):
        instance = cls(TokenizerConfig(top_n, special_tokens, stopwords=set(stopwords), stemmer=stemmer), True)
        instance.fit_tokenizer(column)
        return instance

    def fit_tokenizer(self, column: pd.Series) -> int:
        if not self.__mutable:
            raise RuntimeError("cannot fit an immutable tokenizer")
        
        counter = Counter()
        for text in tqdm(column, total=len(column)):
            tokens = self.__tokenize(text)
            counter.update(tokens)

        top_n = None if self.__config.top_n is None else self.__config.top_n - len(self.__config.special_tokens)
        special_freqs = [(st, counter.pop(st, 0)) for st in self.__config.special_tokens]
        most_common_n = counter.most_common(top_n)
        tokens_w_count = special_freqs + most_common_n

        self.__config.tokens = np.array([tk for tk, _ in tokens_w_count])
        self.__config.token_frequencies = np.array([i for _, i in tokens_w_count])
        self.__config.tokens_to_ids = { tk.item() : inx for inx,tk in enumerate(self.__config.tokens)}
        self.__mutable = False
        return len(tokens_w_count)

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
            tokens = [
                self.__config.stemmer.stemWord(token)
                if token not in self.__config.special_tokens else token
                for token in tokens]
        return tokens