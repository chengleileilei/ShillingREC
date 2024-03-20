import pandas as pd
import numpy as np
from data import dfDataset
from torch.nn import Module

class BaseAttacker(Module):
    _USER = "user"
    _ITEM = "item"
    _RATING = "rating"
    _TIME = "time"
    _LABEL = "label"
    def __init__(self,train_df,config):
        super(BaseAttacker,self).__init__()
        # self.target_id_list = ""

        self.attacker_num = config['attacker_num']
        self.filler_num = config['filler_num']
        self.target_num = config['target_num']

        self.train_df = train_df
        self.dataset = dfDataset(train_df,config)
        self.item_unique = np.sort(self.train_df[self._ITEM].unique())
        self.item_num = self.dataset.n_items # item最大id+1

        self.task_type = config['task_type']

        if self.task_type == 'rating':
            self.all_rating_mean = self.train_df[self._RATING].mean()
            self.all_rating_std = self.train_df[self._RATING].std()
            self.max_rating = self.train_df[self._RATING].max()
            self.min_rating = self.train_df[self._RATING].min()
        
        self.target_id_list = config['target_id_list']

        # if config['target_item_strategy'] == "popular":
        # # 选择流行的item
        #     self.target_id_list = self.train_df[self._ITEM].value_counts().index[:self.target_num]
        # elif config['target_item_strategy'] == "random":
        #     self.target_id_list = np.random.choice(self.item_unique,self.target_num,replace=False) 


        # print("target_id_list",self.target_id_list)
        # config['target_id_list'] = self.target_id_list.tolist()
        print(config)
        # 优化item_num 逻辑

    def get_fake_df(self):
        if self.task_type == 'rating':
            return self.get_explicit_fake_df()
        elif self.task_type == 'ranking':
            return self.get_implicit_fake_df()



    def _fake_profile2df(self, fake_profiles, fake_user_id_list):
        """
        根据fake_profiles（二维列表）和fake_user_id_list构造fake_df
        """
        print("profile2df...")
        print("fake_profiles:\n",fake_profiles)

        user_index_rows,item_id_cols = np.nonzero(fake_profiles)
        user_id_rows = [fake_user_id_list[i] for i in user_index_rows]
        all_cols = self.train_df.columns

        if self.task_type == 'rating':
            ratings = fake_profiles[user_index_rows,item_id_cols]
            fake_df = pd.DataFrame({self._USER: user_id_rows, self._ITEM: item_id_cols, self._RATING: ratings})
            # 构造其余列
            other_cols = list(set(all_cols) - set([self._USER, self._ITEM, self._RATING]))

        elif self.task_type == 'ranking':
            fake_df = pd.DataFrame({self._USER: user_id_rows, self._ITEM: item_id_cols})
            # 构造其余列
            other_cols = list(set(all_cols) - set([self._USER, self._ITEM]))

        # 添加其余列，填充值为train_df对应列的众数
        for col in other_cols:
            fake_df[col] = self.train_df[col].mode()[0]
        # 数据类型对齐
        
        # print("user unique:",fake_df[self._USER].unique())
        # print("rating unique:",fake_df[self._RATING].unique())
        # print("rating count",fake_df[self._RATING].value_counts())
        # print('len for fake_df:',len(fake_df))

        fake_df.dropna(inplace=True) # 去除nan, TODO:后续删除
        fake_df = fake_df.astype(self.train_df.dtypes.to_dict())
        print("generate fake df done!")

        return fake_df


