import torch
import torch.nn as nn
from model import AbstractRecommender, AutoEncoderMixin
from model import xavier_normal_initialization

class CDAE(AbstractRecommender,AutoEncoderMixin):
    def __init__(self,config, dataset):
        super(CDAE,self).__init__(config, dataset)

        self.reg_weight_1 = config['rec_model_p']["reg_weight_1"]
        self.reg_weight_2 = config['rec_model_p']["reg_weight_2"]
        self.loss_type = config['rec_model_p']["loss_type"]
        self.hid_activation = config['rec_model_p']["hid_activation"]
        self.out_activation = config['rec_model_p']["out_activation"]
        self.embedding_size = config['rec_model_p']["embedding_size"]
        self.corruption_ratio = config['rec_model_p']["corruption_ratio"]

        self.build_histroy_items(dataset) # 区分rating和rank任务，rating任务value为分数，rank为0/1

        if self.hid_activation == "sigmoid":
            self.h_act = nn.Sigmoid()
        elif self.hid_activation == "relu":
            self.h_act = nn.ReLU()
        elif self.hid_activation == "tanh":
            self.h_act = nn.Tanh()
        else:
            raise ValueError("Invalid hidden layer activation function")

        if self.out_activation == "sigmoid":
            self.o_act = nn.Sigmoid()
        elif self.out_activation == "relu":
            self.o_act = nn.ReLU()
        else:
            raise ValueError("Invalid output layer activation function")

        self.dropout = nn.Dropout(p=self.corruption_ratio)

        self.h_user = nn.Embedding(self.n_users, self.embedding_size)
        self.h_item = nn.Linear(self.n_items, self.embedding_size)
        self.out_layer = nn.Linear(self.embedding_size, self.n_items)

        # parameters initialization
        self.apply(xavier_normal_initialization)


    def forward(self,bat_user,bat_item):
        h_i = self.dropout(bat_item)
        h_i = self.h_item(h_i)
        h_u = self.h_user(bat_user)
        h = torch.add(h_u, h_i)
        h = self.h_act(h)
        out = self.out_layer(h)
        return out
    

    def calculate_rating_loss(self,bat_users,bat_items,bat_ratings):
        """
        计算评分预测任务的loss, point-wise loss
        """
        x_users = bat_users
        x_items = self.get_rating_matrix(x_users)
        predict = self.forward(x_users, x_items)

        if self.loss_type == "MSE":
            # predict = self.o_act(predict)
            loss_func = nn.MSELoss(reduction="sum")
        elif self.loss_type == "BCE":
            loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        else:
            raise ValueError("Invalid loss_type, loss_type must in [MSE, BCE]")
        loss = loss_func(predict, x_items)
        # l1-regularization
        loss += self.reg_weight_1 * (
            self.h_user.weight.norm(p=1) + self.h_item.weight.norm(p=1)
        )
        # l2-regularization
        loss += self.reg_weight_2 * (
            self.h_user.weight.norm() + self.h_item.weight.norm()
        )
        return loss
    
    def calculate_rank_loss(self,bat_users,bat_pos_items,bat_neg_items):
        """
        计算排序预测任务的loss, pair-wise loss
        """
        x_users = bat_users
        x_items = self.get_rating_matrix(x_users)
        predict = self.forward(x_users, x_items)

        # if self.loss_type == "MSE":
        #     predict = self.o_act(predict)
        #     loss_func = nn.MSELoss(reduction="sum")
        # elif self.loss_type == "BCE":
        #     loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        # else:
        #     raise ValueError("Invalid loss_type, loss_type must in [MSE, BCE]")
        loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        loss = loss_func(predict, x_items)
        # l1-regularization
        loss += self.reg_weight_1 * (
            self.h_user.weight.norm(p=1) + self.h_item.weight.norm(p=1)
        )
        # l2-regularization
        loss += self.reg_weight_2 * (
            self.h_user.weight.norm() + self.h_item.weight.norm()
        )
        return loss

    def predict(self, bat_users,bat_items):
        users = bat_users
        predict_items = bat_items

        items = self.get_rating_matrix(users)
        scores = self.forward(users,items)
        # scores = self.o_act(scores)
        return scores[[torch.arange(len(predict_items)).to(self.device), predict_items]]
    
    def full_sort_predict(self,bat_users):
        """
        预测所有item的评分，用于计算topk
        """ 
        users = bat_users
        items = self.get_rating_matrix(users)
        predict = self.forward(users,items)
        # predict = self.o_act(predict)
        return predict
    











    