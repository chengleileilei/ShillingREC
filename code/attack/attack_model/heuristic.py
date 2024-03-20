from .base_attacker import BaseAttacker
import numpy as np
import pandas as pd
from rectool import getPopularItemId

# def getPopularItemId(item_col, k, exclusion=[]):
#     # 找到k个最热门的item，且与exclusion不重复
#     popular_item_list = item_col.value_counts().index
#     res = []
#     for item in popular_item_list:
#         if item not in exclusion:
#             res.append(item)
#             if len(res) == k:
#                 break
#     return res

class RandomAttack(BaseAttacker):
    
    def __init__(self, train_df,config):
        super().__init__(train_df,config)

    def get_explicit_fake_df(self):
        print("generate fake df...")
        attacker_num = self.attacker_num
        filler_num = self.filler_num
        target_num = self.target_num
        target_id_list = self.target_id_list

        rate = int(attacker_num / target_num) # 每个attack目标的攻击用户数
        fake_profiles = np.zeros(shape=[attacker_num, self.item_num], dtype=float)  # [attacker_num, item_num]
        for i in range(len(target_id_list)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = self.max_rating
        # print('only_target_fake_profiles:\n',fake_profiles)
        filler_pool = list(set(range(self.item_num)) - set(target_id_list)) # 填充物品的id 【0，1，2，3，4，5，6，7，8，9】
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False) # 从filler_pool中随机选择filler_num个物品
        sampled_cols = np.reshape(
            np.array(
                [
                    filler_sampler([filler_pool, filler_num])
                    for _ in range(attacker_num)
                ]
            ),
            (-1),
        )
        sampled_rows = [
            j for i in range(attacker_num) for j in [i] * filler_num
        ]

        sampled_values = np.random.normal(
            loc=self.all_rating_mean,
            scale=self.all_rating_std,
            size=(attacker_num * filler_num),
        )
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > self.max_rating] = self.max_rating
        sampled_values[sampled_values < self.min_rating] = self.min_rating
        fake_profiles[sampled_rows, sampled_cols] = sampled_values

        fake_user_id_start = self.train_df[self._USER].max() + 1
        fake_user_id_list = []

        for i in range(attacker_num):
            fake_user_id_list.extend([fake_user_id_start+i])
        fake_df = self._fake_profile2df(fake_profiles,fake_user_id_list)
        return fake_df
    
    def get_implicit_fake_df(self):
        print("generate fake df...")
        attacker_num = self.attacker_num
        filler_num = self.filler_num
        target_num = self.target_num
        target_id_list = self.target_id_list

        rate = int(attacker_num / target_num) # 每个attack目标的攻击用户数
        fake_profiles = np.zeros(shape=[attacker_num, self.item_num], dtype=float)  # [attacker_num, item_num]
        for i in range(len(target_id_list)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = 1
        # print('only_target_fake_profiles:\n',fake_profiles)
        filler_pool = list(set(range(self.item_num)) - set(target_id_list)) # 填充物品的id 【0，1，2，3，4，5，6，7，8，9】
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False) # 从filler_pool中随机选择filler_num个物品
        sampled_cols = np.reshape(
            np.array(
                [
                    filler_sampler([filler_pool, filler_num])
                    for _ in range(attacker_num)
                ]
            ),
            (-1),
        )
        sampled_rows = [
            j for i in range(attacker_num) for j in [i] * filler_num
        ]

        # sampled_values = np.random.normal(
        #     loc=self.all_rating_mean,
        #     scale=self.all_rating_std,
        #     size=(attacker_num * filler_num),
        # )
        sampled_values = np.ones(attacker_num * filler_num)

        sampled_values = np.round(sampled_values)
        # sampled_values[sampled_values > self.max_rating] = self.max_rating
        # sampled_values[sampled_values < self.min_rating] = self.min_rating
        fake_profiles[sampled_rows, sampled_cols] = sampled_values

        fake_user_id_start = self.train_df[self._USER].max() + 1
        fake_user_id_list = []

        for i in range(attacker_num):
            fake_user_id_list.extend([fake_user_id_start+i])
        fake_df = self._fake_profile2df(fake_profiles,fake_user_id_list)
        return fake_df

class AverageAttack(BaseAttacker):
    def __init__(self,train_df,config):
        super().__init__(train_df,config)

        self.item_mean_dict = {}
        self.item_std_dict = {}
        for item in self.item_unique:
            item_df = self.train_df[self.train_df[self._ITEM] == item]
            self.item_mean_dict[item] = item_df[self._RATING].mean()
            self.item_std_dict[item] = item_df[self._RATING].std()

    def get_explicit_fake_df(self):
        print("generate fake df...")
        attacker_num = self.attacker_num
        filler_num = self.filler_num
        target_num = self.target_num
        target_id_list = self.target_id_list

        rate = int(attacker_num / target_num) # 每个attack目标的攻击用户数
        fake_profiles = np.zeros(shape=[attacker_num, self.item_num], dtype=float)  # [attacker_num, item_num]
        for i in range(len(target_id_list)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = self.max_rating
 
        filler_pool = list(set(range(self.item_num)) - set(target_id_list)) # 填充物品的id 【0，1，2，3，4，5，6，7，8，9】

        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False) # 从filler_pool中随机选择filler_num个物品

        sampled_cols = np.reshape(
            np.array(
                [
                    filler_sampler([filler_pool, filler_num])
                    for _ in range(attacker_num)
                ]
            ),
            (-1),
        )
        sampled_rows = [
            j for i in range(attacker_num) for j in [i] * filler_num
        ]
        # for i in range(attacker_num):
        #     for j in [i]*filler_num:
        #         sampled_rows.append(j)

        # sampled_values = np.random.normal(
        #     loc=self.all_rating_mean,
        #     scale=self.all_rating_std,
        #     size=(attacker_num * filler_num),
        # )
        sampled_values = [
            np.random.normal(
                loc=self.item_mean_dict.get(iid, self.all_rating_mean),
                scale=self.item_std_dict.get(iid, self.all_rating_std),
            )
            for iid in sampled_cols
        ]
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > self.max_rating] = self.max_rating
        sampled_values[sampled_values < self.min_rating] = self.min_rating
        fake_profiles[sampled_rows, sampled_cols] = sampled_values
        # print('finasll fake_profiles',fake_profiles)
        fake_user_id_start = self.train_df[self._USER].max() + 1
        fake_user_id_list = []
        for i in range(attacker_num):
            fake_user_id_list.extend([fake_user_id_start+i])
        fake_df = self._fake_profile2df(fake_profiles,fake_user_id_list)
        return fake_df

class LoveHate(BaseAttacker):
    def __init__(self, train_df,config):
        super().__init__(train_df,config)

    def get_explicit_fake_df(self):
        print("generate fake df...")
        attacker_num = self.attacker_num
        filler_num = self.filler_num
        target_num = self.target_num
        target_id_list = self.target_id_list

        rate = int(attacker_num / target_num) # 每个attack目标的攻击用户数
        fake_profiles = np.zeros(shape=[attacker_num, self.item_num], dtype=float)  # [attacker_num, item_num]
        for i in range(len(target_id_list)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = self.max_rating
        # print('only_target_fake_profiles:\n',fake_profiles)
 
        filler_pool = list(set(range(self.item_num)) - set(target_id_list)) # 填充物品的id 【0，1，2，3，4，5，6，7，8，9】

        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False) # 从filler_pool中随机选择filler_num个物品

        sampled_cols = np.reshape(
            np.array(
                [
                    filler_sampler([filler_pool, filler_num])
                    for _ in range(attacker_num)
                ]
            ),
            (-1),
        )
        sampled_rows = [
            j for i in range(attacker_num) for j in [i] * filler_num
        ]
        sampled_values = np.array([self.min_rating]*attacker_num * filler_num)
        # sampled_values = np.random.normal(
        #     loc=self.all_rating_mean,
        #     scale=self.all_rating_std,
        #     size=(attacker_num * filler_num),
        # )
        sampled_values = np.round(sampled_values)
        sampled_values[sampled_values > self.max_rating] = self.max_rating
        sampled_values[sampled_values < self.min_rating] = self.min_rating
        fake_profiles[sampled_rows, sampled_cols] = sampled_values

        fake_user_id_start = self.train_df[self._USER].max() + 1
        fake_user_id_list = []
        # TODO:fake user 数量存在问题
        for i in range(attacker_num):
            fake_user_id_list.extend([fake_user_id_start+i])
        fake_df = self._fake_profile2df(fake_profiles,fake_user_id_list)
        return fake_df


class BandwagonAttack(BaseAttacker):
    def __init__(self, train_df,config):
        super().__init__(train_df,config)
        selected_item_num = config['attack_model_p']['selected_item_num']
        self.selected_ids = getPopularItemId(self.train_df[self._ITEM], selected_item_num, self.target_id_list)

    # def get_explicit_fake_df(self):
    #     print("generate fake df...")
    #     attacker_num = self.attacker_num
    #     filler_num = self.filler_num
    #     target_num = self.target_num
    #     target_id_list = self.target_id_list

    #     rate = int(attacker_num / target_num) # 每个attack目标的攻击用户数
    #     fake_profiles = np.zeros(shape=[attacker_num, self.item_num], dtype=float)  # [attacker_num, item_num]
    #     for i in range(len(target_id_list)):
    #         fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = self.max_rating
    #     # print('only_target_fake_profiles:\n',fake_profiles)
    #     filler_pool = list(set(range(self.item_num)) - set(target_id_list)) # 填充物品的id 【0，1，2，3，4，5，6，7，8，9】
    #     filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False) # 从filler_pool中随机选择filler_num个物品
    #     sampled_cols = np.reshape(
    #         np.array(
    #             [
    #                 filler_sampler([filler_pool, filler_num])
    #                 for _ in range(attacker_num)
    #             ]
    #         ),
    #         (-1),
    #     )
    #     sampled_rows = [
    #         j for i in range(attacker_num) for j in [i] * filler_num
    #     ]

    #     sampled_values = np.random.normal(
    #         loc=self.all_rating_mean,
    #         scale=self.all_rating_std,
    #         size=(attacker_num * filler_num),
    #     )
    #     sampled_values = np.round(sampled_values)
    #     sampled_values[sampled_values > self.max_rating] = self.max_rating
    #     sampled_values[sampled_values < self.min_rating] = self.min_rating
    #     fake_profiles[sampled_rows, sampled_cols] = sampled_values

    #     fake_user_id_start = self.train_df[self._USER].max() + 1
    #     fake_user_id_list = []

    #     for i in range(attacker_num):
    #         fake_user_id_list.extend([fake_user_id_start+i])
    #     fake_df = self._fake_profile2df(fake_profiles,fake_user_id_list)
    #     return fake_df
    
    def get_implicit_fake_df(self):
        print("generate fake df...")
        attacker_num = self.attacker_num
        filler_num = self.filler_num
        target_num = self.target_num
        target_id_list = self.target_id_list

        rate = int(attacker_num / target_num) # 每个attack目标的攻击用户数
        fake_profiles = np.zeros(shape=[attacker_num, self.item_num], dtype=float)  # [attacker_num, item_num]
        for i in range(len(target_id_list)):
            fake_profiles[i * rate : (i + 1) * rate, target_id_list[i]] = 1
        fake_profiles[:,self.selected_ids] = 1
        # print('only_target_fake_profiles:\n',fake_profiles)
        filler_pool = list(set(range(self.item_num)) - set(target_id_list) - set(self.selected_ids)) # 填充物品的id 【0，1，2，3，4，5，6，7，8，9】
        filler_sampler = lambda x: np.random.choice(x[0], size=x[1], replace=False) # 从filler_pool中随机选择filler_num个物品
        sampled_cols = np.reshape(
            np.array(
                [
                    filler_sampler([filler_pool, filler_num])
                    for _ in range(attacker_num)
                ]
            ),
            (-1),
        )
        sampled_rows = [
            j for i in range(attacker_num) for j in [i] * filler_num
        ]

        # sampled_values = np.random.normal(
        #     loc=self.all_rating_mean,
        #     scale=self.all_rating_std,
        #     size=(attacker_num * filler_num),
        # )
        sampled_values = np.ones(attacker_num * filler_num)

        sampled_values = np.round(sampled_values)
        # sampled_values[sampled_values > self.max_rating] = self.max_rating
        # sampled_values[sampled_values < self.min_rating] = self.min_rating
        fake_profiles[sampled_rows, sampled_cols] = sampled_values

        fake_user_id_start = self.train_df[self._USER].max() + 1
        fake_user_id_list = []

        for i in range(attacker_num):
            fake_user_id_list.extend([fake_user_id_start+i])
        fake_df = self._fake_profile2df(fake_profiles,fake_user_id_list)
        return fake_df


