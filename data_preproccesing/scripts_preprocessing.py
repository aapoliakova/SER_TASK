"""
Perform sentence extraction for each wav and store it in text_data.csv
Unique key example:
"""
import pandas as pd
import os
from tqdm import tqdm
import re

idx = []
sentences = []
for i in tqdm(range(1, 6)):
    path = f'data/IEMOCAP_full_release/Session{i}/dialog/transcriptions'
    filenames = sorted(os.listdir(path))
    useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)
    for file_name in tqdm(filenames):
        file_path = path + "/" + file_name
        with open(file_path, 'r') as f:
            for line in f:
                sentence_id = useful_regex.match(line).group()
                sentence = line.split(':')[-1].strip()

                idx.append(sentence_id)
                sentences.append(sentence.strip())

assert len(idx) == len(sentences), "Indexes quantity should be the same as sentences"
df = pd.DataFrame({
    'id': idx,
    'sentence': sentences
})
df.to_csv('preprocessed_data/preprocessed_text.csv')
# print(df)
#
# d = pd.read_csv('preprocessed_data/preprocessed_text.csv', index_col=0)
# print(d)
