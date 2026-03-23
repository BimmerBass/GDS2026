from sklearn.base import BaseEstimator, TransformerMixin
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import Stemmer
import re

tqdm.pandas()

class TokenTransformer(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            top_n: int = None,
            special_tokens : list[str] = [],
            stopwords: list[str] = [],
            stem: bool = False):
        super().__init__()
        self.top_n = top_n
        self.special_tokens = special_tokens
        self.stopwords : set[str] = stopwords
        self.stem = stem

        self.tokens : np.ndarray = []
        self.token_frequencies : np.ndarray = []
        self.tokens_to_ids : dict[str, int] = {}
        
        # runtime-only cache, not persisted directly
        self.stemmer : Stemmer.Stemmer = None
        self._init_runtime_objects()

    def _init_runtime_objects(self):
        self.stemmer = None
        if self.stem:
            self.stemmer = Stemmer.Stemmer("english")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["stemmer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_runtime_objects()

    def fit(self, X: pd.DataFrame, y = None):
        self.stopwords = set(self.stopwords)
        content_column = X["content"]
        counter = Counter()
        for text in tqdm(content_column, total=len(content_column)):
            tokens = self.__tokenize(text)
            counter.update(tokens)

        top_n = None if self.top_n is None else self.top_n - len(self.special_tokens)
        special_freqs = [(st, counter.pop(st, 0)) for st in self.special_tokens]
        most_common_n = counter.most_common(top_n)
        tokens_w_count = special_freqs + most_common_n

        self.tokens = np.array([tk for tk, _ in tokens_w_count])
        self.token_frequencies = np.array([i for _, i in tokens_w_count])
        self.tokens_to_ids = { tk.item() : inx for inx,tk in enumerate(self.tokens)}
        return self
    
    def size(self) -> int:
        return self.tokens.size

    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        X["content"] = X["content"].progress_apply(self.text_to_ids)
        return X
    
    def text_to_ids(self, text : str) -> np.ndarray:
        terms = self.__tokenize(text)
        tokens = [self.tokens_to_ids[term] for term in terms if term in self.tokens_to_ids]
        return np.array(tokens)

    def __tokenize(self, text: str) -> list[str]:
        text = str(text)
        no_punct = re.sub(r"[^\w<> ]", '', text)    # remove all characters that aren't an alphanumerical one, a space or < / >
        no_ws = re.sub(r"\s+", " ", no_punct) # collapse whitespaces to single ' '
        terms = no_ws.split(" ") # split into terms -> candidate tokens

        trimmed = [term.strip() for term in terms]
        tokens = [term for term in trimmed if term != "" and term not in self.stopwords]
        if self.stemmer != None:
            tokens = [
                self.stemmer.stemWord(token)
                if token not in self.special_tokens else token
                for token in tokens]
        return tokens