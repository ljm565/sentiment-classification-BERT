import torch
import os
import pickle
import pandas as pd



"""
common utils
"""
def load_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_data(base_path):
    if not (os.path.isfile(base_path+'data/processed/data.pkl')):
        print('Processing the Google Play Store Apps review data')
        
        df = pd.read_csv(base_path+'data/raw/reviews.csv')
        content, score = df.loc[:, 'content'], labeling(df.loc[:, 'score'].tolist())
        dataset = [(c, s) for c, s in zip(content, score)]

        with open(base_path+'data/processed/data.pkl', 'wb') as f:
            pickle.dump(dataset, f)


def labeling(score):
    for i, s in enumerate(score):
        if s < 3:
            score[i] = 0
        elif s == 3:
            score[i] = 1
        else:
            score[i] = 2
    return score


def label_mapping(score):
    if score == 0:
        return "negative"
    elif score == 1:
        return "mediocre"
    return "positive"
      

def make_dataset_path(base_path):
    dataset_path = base_path + 'data/processed/data.pkl'
    return dataset_path


def save_checkpoint(file, model, optimizer):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, file)
    print('model pt file is being saved\n')