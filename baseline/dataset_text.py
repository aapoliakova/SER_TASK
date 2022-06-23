import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

"""
Return transformed wav file and labels
"""
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"





class IEMOCAMP_TEXT(Dataset):
    def __init__(self, dataset_path, tokenizer=None):
        self.data = pd.read_csv(dataset_path)
        self.transform = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        y = self.data['labels'][index]
        X = self.data['sentence'][index]
        return {'sentences': X, 'label': y}


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    usd = IEMOCAMP_TEXT(dataset_path='preprocessed_data/text_train.csv',
                        tokenizer=tokenizer,
                        )
    print(f"There are {len(usd)} samples in the dataset.")
    text, label = usd[0]  # (1, 64, 64)
    print(label, text)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_data = IEMOCAMP_TEXT(dataset_path='preprocessed_data/text_train.csv',
                               tokenizer=tokenizer,
                               )


    train_dataloader = DataLoader(train_data, batch_size=64)
    batch = next(iter(train_dataloader))
    print(type(batch['sentences']))