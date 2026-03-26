from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib
from tqdm import tqdm
import os
tqdm.pandas()

class AdvancedModel:
    @classmethod
    def load(cls, path):
        model = cls()
        model.tokenizer = joblib.load(path + "/tokenizer.joblib")
        model.vectorizer = joblib.load(path + "/vectorizer.joblib")
        model.model = joblib.load(path + "/model.joblib")
        return model

    # Needs to be a proper function since lambda expression doesn't work with joblib apparently
    @classmethod
    def intarr_to_strarr(cls, intarr):
        return [str(id) for id in intarr]


    @classmethod
    def create(cls, tokenizer : str, min_ngram : int, max_ngram : int, max_features : int):
        model = cls()
        model.tokenizer = joblib.load(tokenizer)
        model.vectorizer = TfidfVectorizer(
            #token_pattern=r"\d+",
            tokenizer=AdvancedModel.intarr_to_strarr,
            ngram_range=(min_ngram, max_ngram),
            max_features=max_features,
            lowercase=False
        )
        model.model = LogisticRegression(max_iter=500, C=10, class_weight='balanced', solver='sag', random_state=0)
        return model

    def __init__(self):
        pass

    def fit(self, content : pd.Series, y : pd.Series):
        print("Tokenizing content...")
        tokens = content.progress_map(self.tokenizer.text_to_ids)

        print("Fitting vectorizer...")
        X = self.vectorizer.fit_transform(tokens)

        print("Fitting model...")
        self.model.fit(X, y)

        print("Fitting complete.")

    def predict(self, content : pd.Series):
        print("Tokenizing content...")
        tokens = content.progress_map(self.tokenizer.text_to_ids)
        return self.model.predict(self.vectorizer.transform(tokens))

    def save(self, path : str, compress : int = 0):
        if not os.path.isdir(path):
            os.mkdir(path)

        joblib.dump(self.tokenizer, path + "/tokenizer.joblib", compress=compress)
        joblib.dump(self.vectorizer, path + "/vectorizer.joblib", compress=compress)
        joblib.dump(self.model, path + "/model.joblib", compress=compress)
