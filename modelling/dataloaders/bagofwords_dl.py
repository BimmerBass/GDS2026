from preprocessing.tokenization import Tokenizer
from torch.utils.data import Dataset
from typing import Tuple
from tqdm import tqdm
import pandas as pd
import torch

class BagofWordsDataset(Dataset):
    def __init__(self, ds_path : str, vocab_path : str):
        super().__init__()
        df = pd.read_csv(ds_path, usecols=["content", "type"])
        df["type"] = df["type"].astype("category")
        # despite the name, unreliable is defined as "Sources that may be reliable but whose contents require further verification."
        reliables = df["type"].isin(["reliable", "political", "clickbait", "bias", "unreliable"])
        
        self.tokenizer = Tokenizer.from_json(vocab_path, False)
        self.__labels = torch.tensor(reliables.astype(int).values).float()

        print(f"tokenizing {len(df)} documents")
        self.__documents = [self.tokenizer.text_to_ids(text) for text in tqdm(df["content"])]
        assert len(self.__documents) == len(self.__labels)


    def __getitem__(self, inx : int) -> Tuple[torch.Tensor, int]:
        tokens = torch.tensor(self.__documents[inx], dtype=torch.int64)
        bow = torch.bincount(tokens, minlength=self.tokenizer.size()).float()
        return bow, self.__labels[inx]
    
    def __len__(self) -> int:
        return len(self.__documents)