import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ACDR(nn.Module):
    def __init__(self, hidden_size, user_size, item_size, num_classes=3, dropout=0, gpu=False):
        super(ACDR, self).__init__()
        self.hidden_size = hidden_size
        self.user_size = user_size
        self.item_size = item_size
        self.dropout = dropout
        self.gpu = gpu

        # user & item embeddings
        self.user_embedding_layer = nn.Embedding(user_size, hidden_size)
        self.item_embedding_layer = nn.Embedding(item_size, hidden_size)
        
        # recommendation layer
        self.linear = nn.Linear(2*hidden_size, hidden_size)
        self.affine_linear = nn.Linear(hidden_size, 1)
        
        # discrimination layer
        self.discriminate_linear = nn.Linear(hidden_size, num_classes)

        # domain embedding generator
        self.domain_1_linear = nn.Linear(hidden_size, hidden_size)
        self.domain_2_linear = nn.Linear(hidden_size, hidden_size)
        self.domain_3_linear = nn.Linear(hidden_size, hidden_size)
        
        self.initialize_param()
        
    def initialize_param(self, initrange=0.1):
        
        # initialize embedding matrix weights
        self.user_embedding_layer.weight.data.uniform_(-initrange, initrange)
        self.item_embedding_layer.weight.data.uniform_(-initrange, initrange)
        
        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)
        self.affine_linear.weight.data.uniform_(-initrange, initrange)
        self.affine_linear.bias.data.fill_(0)
        
        self.discriminate_linear.weight.data.uniform_(-initrange, initrange)
        self.discriminate_linear.bias.data.fill_(0)
        
        self.domain_1_linear.weight.data.uniform_(-initrange, initrange)
        self.domain_1_linear.bias.data.fill_(0)
        self.domain_2_linear.weight.data.uniform_(-initrange, initrange)
        self.domain_2_linear.bias.data.fill_(0)
        self.domain_3_linear.weight.data.uniform_(-initrange, initrange)
        self.domain_3_linear.bias.data.fill_(0)
        
    def predict(self, user_id, item_id, label):
        user_embedding = self.user_embedding_layer(user_id)
        if label == 0:
            domain_user_embedding = self.domain_1_linear(user_embedding)
        elif label == 1:
            domain_user_embedding = self.domain_2_linear(user_embedding)
        elif label == 2:
            domain_user_embedding = self.domain_3_linear(user_embedding)
        item_embedding = self.item_embedding_layer(item_id)
        vector = torch.cat([domain_user_embedding, item_embedding], dim=-1)
        vector = vector.float()
        vector = self.linear(vector)
        vector = torch.nn.Dropout(p=0.1)(vector)
        vector = torch.nn.ReLU()(vector)
        #vector = torch.nn.BatchNorm1d()(vector)
        rating = self.affine_linear(vector)
        #rating = self.logistic(rating)
        return rating
    
    def discriminate(self, user_id, label):
        user_embedding = self.user_embedding_layer(user_id)
        if label == 0:
            domain_user_embedding = self.domain_1_linear(user_embedding)
        elif label == 1:
            domain_user_embedding = self.domain_2_linear(user_embedding)
        elif label == 2:
            domain_user_embedding = self.domain_3_linear(user_embedding)
        classification = self.discriminate_linear(domain_user_embedding)
        return classification
    
def evaluate(model, evaluate_data, epoch_id, label, gpu=False):
    model.eval()
    user_id, item_id, rating = Variable(evaluate_data[0]), Variable(evaluate_data[1]), Variable(evaluate_data[2])
    rating = rating.float()
    if gpu:
        model.cuda()
        rating = rating.cuda()
        user_id = user_id.cuda()
        item_id = item_id.cuda()
    ratings_pred = model.predict(user_id, item_id, label)
    ratings_pred = ratings_pred.detach().numpy()
    rating = rating.detach().numpy()
    rmse = math.sqrt(mean_squared_error(ratings_pred, rating))
    mae = mean_absolute_error(ratings_pred, rating)
        
    user_pool = list(set(user_id))
    recommend, precision, recall = [], [], []
    for index in range(len(user_pool)):
        recommend.append((user_id[index],item_id[index],rating[index],ratings_pred[index]))
    for user in user_pool:
            user_ratings = [x for x in recommend if x[0]==user]
            user_ratings.sort(key=lambda x:x[3], reverse=True)
            n_rel = sum((true_r >= 0.5) for (_, _, true_r, _) in user_ratings)
            n_rec_k = sum((est >= 0.5) for (_, _, _, est) in user_ratings)
            n_rel_and_rec_k = sum(((true_r >= 0.5) and (est >= 0.5))
                            for (_, _, true_r, est) in user_ratings)
            precision.append(n_rel_and_rec_k / n_rec_k if n_rec_k!=0 else 1)
            recall.append(n_rel_and_rec_k / n_rel if n_rel!=0 else 1)
    precision = np.mean(precision)
    recall = float(np.mean(recall))
    print('[Evluating Epoch {}] Domain {} RMSE = {:.4f}, MAE = {:.4f}, Precision = {:.4f}, Recall = {:.4f}'.format(epoch_id, label, rmse, mae, precision, recall))
    
    return rmse, mae, precision, recall