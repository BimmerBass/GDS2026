from nltk.stem import PorterStemmer, SnowballStemmer, StemmerI
from tokenizer import Tokenizer, TokenizerConfig
from nltk.corpus import stopwords
from pathlib import Path
import pandas as pd
import argparse
import json
import nltk
import yaml

nltk.download('stopwords')
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()
    return args

def load_data(config : dict[str,str]) -> pd.DataFrame:
    print(f"Reading CSV {config['file']}")
    print(f"- Loading columns: [{', '.join(config['usecols'])}]")
    df = pd.read_csv(config["file"], usecols=config['usecols'])
    return df

def build_tokenizer(column: pd.DataFrame, config : dict[str, str]) -> Tokenizer:
    tk_stopwords = stopwords.words("english") if config["remove_stopwords"] else []
    stemmer: StemmerI = None
    match config["stemming"]:
        case "porter":
            stemmer = PorterStemmer()
        case "snowball":
            stemmer = SnowballStemmer()
        case _:
            stemmer = None
    top_n = config["top_n"] if config["top_n"] is not None else 1000000000
    print("Building tokenizer")
    return Tokenizer.create_and_fit(column, top_n, ["<DATE>", "<NUM>", "<URL>", "<EMAIL>"], tk_stopwords, stemmer)


def save_tokenizer(config : TokenizerConfig, filepath : str) -> None:
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, ensure_ascii=False, indent=4)


if __name__=="__main__":
    args = parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    df = load_data(config["load_data"])
    tokenizer = build_tokenizer(df["content"], config["build_tokenizer"])
    save_tokenizer(tokenizer.get_config(), config["save_data"]["filename"])