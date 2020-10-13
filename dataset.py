from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable

class TrainDataset(Dataset):
    def __init__(self, data):
        max_rating = data.rate.max()
        data['rate'] = data.rate * 1.0 / max_rating
        self.user_pool = set(data['userId'].unique())
        self.item_pool = set(data['itemId'].unique())
        users, items, ratings, labels = [], [], [], []
        cut = 4 * len(data) // 5
        data = data.sample(frac=1)
        train = data[:cut]
        train = train.sort_values(by=['label'])
        
        for row in train.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rate))
            labels.append(row.label)
        
        self.user_tensor = torch.LongTensor(users)
        self.item_tensor = torch.LongTensor(items)
        self.rating_tensor = torch.FloatTensor(ratings)
        self.label_tensor = labels

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.rating_tensor[index], self.label_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)

def TestDataset(data, label):
    max_rating = data.rate.max()
    data['rate'] = data.rate * 1.0 / max_rating
    users, items, ratings, labels = [], [], [], []
    data = data.sample(frac=1)
    cut = 4 * len(data) // 5
    test = data[cut:]
    test = test[test['label']==label] 
    for row in test.itertuples():
        users.append(int(row.userId))
        items.append(int(row.itemId))
        ratings.append(float(row.rate))
        labels.append(row.label)     
    user_tensor = torch.LongTensor(users)
    item_tensor = torch.LongTensor(items)
    rating_tensor = torch.FloatTensor(ratings)
    return user_tensor, item_tensor, rating_tensor, label
