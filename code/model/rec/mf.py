import torch
import torch.nn as nn
from model import AbstractRecommender
from model import xavier_normal_initialization

class MF(AbstractRecommender):
    def __init__(self,config, dataset):
        super(MF,self).__init__(config, dataset)

        self.embedding_size = config["rec_model_p"]["embedding_size"]
        self.dropout_prob = config["rec_model_p"]["dropout_prob"]
        
        self.user_embedding = nn.Embedding(self.n_users,self.embedding_size)
        self.user_bias = nn.Embedding(self.n_users,1)
        self.item_embedding = nn.Embedding(self.n_items,self.embedding_size)
        self.item_bias = nn.Embedding(self.n_items,1)
        self.dropout_layer = nn.Dropout(p=self.dropout_prob)

        # print(self.user_embedding.weight.shape)
        # print(self.item_embedding.weight.shape)
        self.apply(xavier_normal_initialization)


    def forward(self,bat_user,bat_item):
        bat_user_embedding = self.user_embedding(bat_user)
        bat_item_embedding = self.item_embedding(bat_item)
        return bat_user_embedding, bat_item_embedding


    def calculate_rating_loss(self,bat_users,bat_items,bat_ratings):
        """
        计算评分预测任务的loss, point-wise loss
        """
        bat_user_embedding, bat_item_embedding = self.forward(bat_users,bat_items)
        bat_user_embedding_bais = self.user_bias(bat_users).squeeze()
        bat_item_embedding_bais = self.item_bias(bat_items).squeeze()
        pre_rating= self.dropout_layer(torch.sum(bat_user_embedding*bat_item_embedding,1)+bat_user_embedding_bais+bat_item_embedding_bais)
        loss = torch.nn.MSELoss()(pre_rating,bat_ratings)
        return loss
    
    def calculate_rank_loss(self,bat_users,bat_pos_items,bat_neg_items):
        """
        计算排序预测任务的loss, pair-wise loss
        """
        bat_user_embedding, bat_pos_item_embedding = self.forward(bat_users,bat_pos_items)
        bat_user_embedding, bat_neg_item_embedding = self.forward(bat_users,bat_neg_items)
        bat_user_embedding_bais = self.user_bias(bat_users).squeeze()
        bat_pos_item_embedding_bais = self.item_bias(bat_pos_items).squeeze()
        bat_neg_item_embedding_bais = self.item_bias(bat_neg_items).squeeze()
        pos_score = self.dropout_layer(torch.sum(bat_user_embedding*bat_pos_item_embedding,1)+bat_user_embedding_bais+bat_pos_item_embedding_bais)
        neg_score = self.dropout_layer(torch.sum(bat_user_embedding*bat_neg_item_embedding,1)+bat_user_embedding_bais+bat_neg_item_embedding_bais)
        loss = torch.sum(-torch.log(torch.sigmoid(pos_score-neg_score)))
        return loss

    def predict(self, bat_users,bat_items):
        bat_user_embedding, bat_item_embedding = self.forward(bat_users,bat_items)
        bat_user_embedding_bais = self.user_bias(bat_users).squeeze()
        bat_item_embedding_bais = self.item_bias(bat_items).squeeze()
        pre_score= self.dropout_layer(torch.sum(bat_user_embedding*bat_item_embedding,1)+bat_user_embedding_bais+bat_item_embedding_bais)
        return pre_score
    
    def full_sort_predict(self,bat_users):
        """
        预测所有item的评分，用于计算topk
        """ 
        bat_user_embedding = self.user_embedding(bat_users) + self.user_bias(bat_users)
        all_item_embedding = self.item_embedding.weight + self.item_bias.weight
        pre_score = torch.matmul(bat_user_embedding,all_item_embedding.transpose(0,1))
        # print("pre_score shape:{}".format(pre_score.shape))
        # full_item_rank = torch.argsort(pre_score,descending=True)
        # print("full_item_rank shape:{}".format(full_item_rank.shape))
        return pre_score
    











    