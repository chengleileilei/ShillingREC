import torch
import torch.nn as nn

class AbstractRecommender(nn.Module):
    def __init__(self,config,datset):
        super(AbstractRecommender,self).__init__()
        # print('AbstractRecommender init')
        self.dataset = datset
        self.n_users = datset.n_users 
        self.n_items = datset.n_items
        if config['task_type'] == 'rating':
            self.train_rating_min = datset.train_rating_min
            self.train_rating_max = datset.train_rating_max
        self.logger = None
        self.evluator = None

        self.device = config["device"]
    

    def forward(self):
        """
        前向传播
        对于graph类型模型行，计算交互矩阵与当前全部user item embedding的乘积，输出新的全部user item embedding
        对于非graph类型模型，计算输入user和item的embedding
        """
        raise NotImplementedError


    def caculate_rating_loss(self,bat_users,bat_items,bat_ratings):
        """
        计算评分预测任务的loss, point-wise loss
        """
        raise NotImplementedError
    
    def caculate_ranking_loss(self,bat_users,bat_pos_items,bat_neg_items):
        """
        计算排序预测任务的loss, pair-wise loss
        """
        raise NotImplementedError
    
    def predict(self,bat_users,bat_items):
        """
        预测bat_users对bat_items的评分[0,1]
        """
        raise NotImplementedError
    
    def predict_rating(self,bat_users,bat_items):
        """
        根据rating区间预测bat_users对bat_items的评分[self.train_rating_min,self.train_rating_max]
        """
        # v = self.predict(bat_users,bat_items)
        # return v*(self.train_rating_max-self.train_rating_min)+self.train_rating_min
        raise NotImplementedError
        
    
    def full_sort_predict(self,bat_users):
        """
        预测所有item的评分，用于计算topk
        """
        raise NotImplementedError



