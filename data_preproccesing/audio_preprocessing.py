import numpy as np
import pandas as pd
import torch

path = 'preprocessed_data/preprocessed_audio.csv'
audio_df = pd.read_csv(path)
d = audio_df.path.str.split('/')
ids = []
for line in d:
    ids.append(line[-1][:-4])

audio_df['id'] = ids

classes = {
    'ang': 0,
    'exc': 1,
    'hap': 1,
    'sad': 2,
    'neu': 3,
    'fru': 4,
    'xxx': 10,
    'sur': 10,
    'oth': 10,
    'fea': 10,
    'dis': 10
}

# Create full data set with text and audio transcripts
text_df = pd.read_csv('preprocessed_data/preprocessed_text.csv', index_col=0)
full_dataset = pd.merge(text_df, audio_df, on='id')
full_dataset['labels'] = full_dataset.emotion.apply(lambda x: classes[x])
full_dataset.to_csv('preprocessed_data/full_dataset.csv', index=False)

"""
4 labels extraction: Angry Happy Sad Nuteral
angry : 0 ang(1103)
happy : 1 exc(1041), hap(595)
sad : 2 sad(1084)
neutral : 3 neu(1708)
frustrated : 4 fru(1849)
Whole dataset is 7380

"""
# anger, happiness, excitement, sadness, frustration, fear, surprise, other and neutral state

"""
For a dev/test set we will leave session 5
"""
df = full_dataset.loc[full_dataset['emotion'].isin(['ang', 'exc', 'hap', 'sad', 'neu', 'fru'])]
df_audio = df[['session', 'path', 'labels']]


df_test_audio = df_audio.loc[df_audio['session'] == 5]
df_train_audio = df_audio.loc[df_audio['session'] != 5]
df_train_audio.to_csv('preprocessed_data/audio_train.csv', index=False)
df_test_audio.to_csv('preprocessed_data/audio_test.csv', index=False)


df_text = df[['session', 'sentence', 'labels']]
df_test_text = df_text.loc[df_text['session'] == 5]
df_train_text = df_text.loc[df_text['session'] != 5]
df_train_text.to_csv('preprocessed_data/text_train.csv', index=False)
df_test_text.to_csv('preprocessed_data/text_test.csv', index=False)



