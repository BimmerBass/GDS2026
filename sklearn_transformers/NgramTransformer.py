from sklearn.base import BaseEstimator, TransformerMixin
from .PrefitTransformer import PrefitTransformer
from collections import Counter
from typing import Sequence
from typing import Tuple
from tqdm import tqdm
import pandas as pd
import numpy as np
import Stemmer
import re

tqdm.pandas()

class NgramTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer : PrefitTransformer, ngram_range : Tuple[int,int], top_n : int = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.ngram_range = ngram_range
        self.top_n = top_n
        self.ngrams : list[Tuple] = []
        self.ngram_frequencies : list[int] = []
        self.ngram_to_id : dict[Tuple,int] = {}

        n_min, n_max = self.ngram_range
        if n_min < 1 or n_max < 1 or n_min > n_max:
            raise ValueError(f"invalid ngram range, {ngram_range}")
    
    def fit(self, X : pd.DataFrame, y=None):
        id_lists : list[list[int]] = self.tokenizer.transform(X)
        counter = Counter()
        for idlist in tqdm(id_lists):
            idlist = [id.item() for id in idlist]
            ngrams = self.__get_ngrams(idlist)
            counter.update(ngrams)

        ngrams_w_count = counter.most_common(self.top_n)
        self.ngrams = [ngram for ngram, _ in ngrams_w_count]
        self.ngram_frequencies = [f for _, f in ngrams_w_count]
        self.ngram_to_id = {ngram : inx for inx,ngram in enumerate(self.ngrams)}

        self.is_fitted_ = True
        return self
    
    def transform(self, X:pd.DataFrame) -> Sequence[list[int]]:
        return X["content"].progress_apply(self.text_to_ids)

    def text_to_ids(self, text : str) -> list[int]:
        unigram_ids : list[int] = self.tokenizer.transformer.text_to_ids(text)
        ngrams = self.__get_ngrams(unigram_ids)
        return [self.ngram_to_id[ngram] for ngram in ngrams if ngram in self.ngram_to_id]

    def __get_ngrams(self, ids : list[int]) -> list[Tuple]:
        n_min,n_max = self.ngram_range
        ngrams = []
        for n in range(n_min, n_max+1):
            ngrams.extend(self.__range_ngrams(ids, n))
        return ngrams
    
    def __range_ngrams(self, ids : list[int], n : int) -> list[Tuple]:
        if len(ids) < n:
            return []
        return [tuple(ids[i:i+n]) for i in range(len(ids) - n + 1)]