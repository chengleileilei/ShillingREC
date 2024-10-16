import torch
import torch.nn as nn
from torch.nn.init import normal_

from model import AbstractRecommender
from model import xavier_normal_initialization,xavier_uniform_initialization
from model import BPRLoss, EmbLoss
from model import MLPLayers


class NeuMF(AbstractRecommender):

    def __init__(self, config, dataset):
        super(NeuMF, self).__init__(config, dataset)

        # load dataset info
        # self.LABEL = config["rec_model_p"]["LABEL_FIELD"]

        # load parameters info
        self.mf_embedding_size = config["rec_model_p"]["mf_embedding_size"]
        self.mlp_embedding_size = config["rec_model_p"]["mlp_embedding_size"]
        self.mlp_hidden_size = config["rec_model_p"]["mlp_hidden_size"]
        self.dropout_prob = config["rec_model_p"]["dropout_prob"]
        self.mf_train = config["rec_model_p"]["mf_train"]
        self.mlp_train = config["rec_model_p"]["mlp_train"]
        self.use_pretrain = config["rec_model_p"]["use_pretrain"]
        self.mf_pretrain_path = config["rec_model_p"]["mf_pretrain_path"]
        self.mlp_pretrain_path = config["rec_model_p"]["mlp_pretrain_path"]

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.mlp_layers = MLPLayers(
            [2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob
        ) # in:[2*mlp_embedding_size] out:[mlp_hidden_size[-1] 
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(
                self.mf_embedding_size + self.mlp_hidden_size[-1], 1
            )
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def load_pretrain(self):
        r"""A simple implementation of loading pretrained parameters."""
        mf = torch.load(self.mf_pretrain_path, map_location="cpu")
        mlp = torch.load(self.mlp_pretrain_path, map_location="cpu")
        mf = mf if "state_dict" not in mf else mf["state_dict"]
        mlp = mlp if "state_dict" not in mlp else mlp["state_dict"]
        self.user_mf_embedding.weight.data.copy_(mf["user_mf_embedding.weight"])
        self.item_mf_embedding.weight.data.copy_(mf["item_mf_embedding.weight"])
        self.user_mlp_embedding.weight.data.copy_(mlp["user_mlp_embedding.weight"])
        self.item_mlp_embedding.weight.data.copy_(mlp["item_mlp_embedding.weight"])

        mlp_layers = list(self.mlp_layers.state_dict().keys())
        index = 0
        for layer in self.mlp_layers.mlp_layers:
            if isinstance(layer, nn.Linear):
                weight_key = "mlp_layers." + mlp_layers[index]
                bias_key = "mlp_layers." + mlp_layers[index + 1]
                assert (
                    layer.weight.shape == mlp[weight_key].shape
                ), f"mlp layer parameter shape mismatch"
                assert (
                    layer.bias.shape == mlp[bias_key].shape
                ), f"mlp layer parameter shape mismatch"
                layer.weight.data.copy_(mlp[weight_key])
                layer.bias.data.copy_(mlp[bias_key])
                index += 2

        predict_weight = torch.cat(
            [mf["predict_layer.weight"], mlp["predict_layer.weight"]], dim=1
        )
        predict_bias = mf["predict_layer.bias"] + mlp["predict_layer.bias"]

        self.predict_layer.weight.data.copy_(predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        # in: user:[batch_size] item:[batch_size]
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(
                torch.cat((user_mlp_e, item_mlp_e), -1)
            )  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.predict_layer(torch.cat((mf_output, mlp_output), -1)) # [batch_size, 1]
        elif self.mf_train:
            output = self.predict_layer(mf_output)
        elif self.mlp_train:
            output = self.predict_layer(mlp_output)
        else:
            raise RuntimeError(
                "mf_train and mlp_train can not be False at the same time"
            )
        return output.squeeze(-1)

    # def calculate_loss(self, interaction):
    #     user = interaction[self.USER_ID]
    #     item = interaction[self.ITEM_ID]
    #     label = interaction[self.LABEL]

    #     output = self.forward(user, item)
    #     return self.loss(output, label)
    
    def calculate_rating_loss(self,bat_users,bat_items,bat_ratings):
        pre_rating = self.forward(bat_users,bat_items)
        loss = torch.nn.MSELoss()(pre_rating,bat_ratings)
        return loss
    
    def calculate_rank_loss(self, bat_users, bat_pos_items, bat_neg_items):
        pos_score = self.forward(bat_users,bat_pos_items)
        neg_score = self.forward(bat_users,bat_neg_items)
        loss = torch.sum(-torch.log(torch.sigmoid(pos_score-neg_score)))
        return loss
    
    
    def predict(self, bat_users,bat_items):

        predict = self.forward(bat_users, bat_items)

        return predict
    

    def full_sort_predict(self,bat_users):
        # 利用forward输出为[batch_size, n_items]矩阵，forward的输入要求user和item的形状相同，所以这里需要对user进行扩展

        all_items = torch.arange(self.n_items).to(bat_users.device).repeat(len(bat_users)) # [batch_size*n_items]
        bat_users = bat_users.repeat_interleave(self.n_items) # [batch_size*n_items]
        predict = self.forward(bat_users, all_items)
        predict = predict.view(-1,self.n_items)
        return predict






    #     predict = self.forward(bat_users, torch.arange(self.n_items).to(bat_users.device))
    #     return predict

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain."""
        if self.mf_train and not self.mlp_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)
