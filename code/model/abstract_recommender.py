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




class AutoEncoderMixin(object):
    """This is a common part of auto-encoders. All the auto-encoder models should inherit this class,
    including CDAE, MacridVAE, MultiDAE, MultiVAE, RaCT and RecVAE.
    The base AutoEncoderMixin class provides basic dataset information and rating matrix function.
    """

    def build_histroy_items(self, dataset):
        self.history_item_id, self.history_item_value, _ = dataset.history_item_matrix() # 训练集的matrix
        self.history_item_id = self.history_item_id.to(self.device)
        self.history_item_value = self.history_item_value.to(self.device)

    def get_rating_matrix(self, user):
        r"""Get a batch of user's feature with the user's id and history interaction matrix.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The user's feature of a batch of user, shape: [batch_size, n_items]
        """
        # Following lines construct tensor of shape [B,n_items] using the tensor of shape [B,H]
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).repeat_interleave(
            self.history_item_id.shape[1], dim=0
        )
        rating_matrix = torch.zeros(1, device=self.device).repeat(
            user.shape[0], self.n_items
        )
        rating_matrix.index_put_(
            (row_indices, col_indices), self.history_item_value[user].flatten()
        )
        return rating_matrix
