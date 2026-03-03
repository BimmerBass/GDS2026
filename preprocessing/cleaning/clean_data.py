from tqdm import tqdm
import pandas as pd
import numpy as np
import yaml
import re
import argparse
from pathlib import Path


tqdm.pandas()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    args = parser.parse_args()
    return args

def load_data(filepath : str) -> pd.DataFrame:
    print(f"Reading CSV {filepath}")
    df = pd.read_csv(filepath, usecols=["id", "domain", "type", "url", "content", "title"])
    return df

def save_data(df: pd.DataFrame, filepath: str) -> None:
    print(f"Saving cleaned data to {filepath}")
    df.to_csv(filepath, index=False)



def clean_text(text : str, config:dict[str,str]) -> str:
    text = str(text)

    if config["lower"]:
        text = text.lower()
    for regex in config["regexes"]:
        text = re.sub(regex, config["regexes"][regex], text)
    if config["collapse_whitespace"]:
        text = re.sub(r"\s+", " ", text)
    return text


def clean_data(df: pd.DataFrame, config : dict[str, str]) -> pd.DataFrame:
    print(f"Cleaning {len(df)} rows")
    df["type"] = df["type"].astype("category")

    print("\nCleaning content")
    df["content"] = df["content"].progress_apply(clean_text, config=config)

    print("\nCleaning title")
    df["title"] = df["title"].progress_apply(clean_text, config=config)
    return df

if __name__=="__main__":
    args = parse_args()
    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)

    df = load_data(config["load_data"]["file"])
    cleaned = clean_data(df, config["clean_data"])
    save_data(cleaned, config["save_data"]["filename"])