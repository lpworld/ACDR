from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd

from model import ACDR, evaluate
from dataset import TrainDataset, TestDataset

batch_size = 32
hidden_size = 128
epoch = 10
lr = 0.01
gpu=False

book = pd.read_csv('book_test.csv')
movie = pd.read_csv('movie_test.csv')
music = pd.read_csv('music_test.csv')
# add the domain category labels: book-> 0, movie-> 1, music-> 2
book['label'] = [0]*len(book)
movie['label'] = [1]*len(movie)
music['label'] = [2]*len(music)
data = pd.concat([book, movie, music],sort=True)
user_id = data[['user_id']].drop_duplicates().reindex()
user_id['userId'] = np.arange(len(user_id))
data = pd.merge(data, user_id, on=['user_id'], how='left')
element_id = data[['element_id']].drop_duplicates().reindex()
element_id['itemId'] = np.arange(len(element_id))
data = pd.merge(data, element_id, on=['element_id'], how='left')
data.index = range(len(data))
data = data[['userId','itemId','rate','rate_date','label']]
train_loader = DataLoader(TrainDataset(data=data), batch_size=batch_size, shuffle=False)
evaluate_data_1 = TestDataset(data, label=0)
evaluate_data_2 = TestDataset(data, label=1)
evaluate_data_3 = TestDataset(data, label=2)
user_size = len(set(data['userId']))
item_size = len(set(data['itemId']))

model = ACDR(hidden_size, user_size, item_size)
for epoch_id in range(epoch):
    model.train()
    total_loss = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    predict_crit = torch.nn.MSELoss()
    discriminate_crit = torch.nn.CrossEntropyLoss()
    for batch in train_loader:
        optimizer.zero_grad()
        user_id, item_id, rating, label = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), batch[3][0]
        rating = rating.float()
        label_target = batch[3]
        if gpu:
            model.cuda()
            rating = rating.cuda()
            user_id = user_id.cuda()
            item_id = item_id.cuda()
            label_target = label_target.cuda()
        ratings_pred = model.predict(user_id, item_id, label)
        predict_loss = predict_crit(ratings_pred.squeeze(1), rating)
        label_pred = model.discriminate(user_id, label)
        discriminate_loss = discriminate_crit(label_pred, label_target)
        loss = predict_loss + discriminate_loss
        loss.backward()
        optimizer.step()
        total_loss += loss
    print(total_loss)
    evaluate(model, evaluate_data_1, epoch_id, label=0, gpu=gpu)
    evaluate(model, evaluate_data_2, epoch_id, label=1, gpu=gpu)
    evaluate(model, evaluate_data_3, epoch_id, label=2, gpu=gpu)


