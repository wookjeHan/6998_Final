import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def get_dataset(split):
    assert split in ['train', 'validation'], "split shoud be train or val"
    ds = load_dataset("osunlp/TravelPlanner", split)[split]
    return ds    

def get_dataloader(split, batch_size, shuffle):
    """"
    split -> str (train / val)
    batch_size -> int
    shuffle -> bool (True / False)
    """
    assert split in ['train', 'validation'], "split shoud be train or val"
    ds = load_dataset("osunlp/TravelPlanner", split)[split]
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dataloader
    