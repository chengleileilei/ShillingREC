__all__ = ["Preprocessor"]

import os
import pandas as pd 
import numpy as np
from datetime import datetime
# from rectool import Configurator
from tqdm import tqdm


class Preprocessor(object):
    _USER = "user"
    _ITEM = "item"
    _RATING = "rating"
    _TIME = "time"
    _TIME_STAMP = "time_stamp"

    def __init__(self, config, raw_df):
        self.config = config
        self.raw_df = raw_df
        self.all_data = None

    def _drop_duplicates(self):
        pass

    def _filter_data(self,user_min=0,item_min=0):
        self._filter_item(item_min)
        self._filter_user(user_min)

    def _filter_user(self,user_min=0): 
        if user_min > 0:
            print("filtering users...")
            user_count = self.all_data[self._USER].value_counts(sort=False)
            filtered_idx = self.all_data[self._USER].map(lambda x: user_count[x] >= user_min)
            self.all_data = self.all_data[filtered_idx]

    def _filter_item(self,item_min=0):
        if item_min > 0:
            print("filtering items...")
            item_count = self.all_data[self._ITEM].value_counts(sort=False)
            filtered_idx = self.all_data[self._ITEM].map(lambda x: item_count[x] >= item_min)
            self.all_data = self.all_data[filtered_idx]
    
    def _map_id(self):
        map_user = {}
        map_item = {}
        user_set = set(self.all_data[self._USER])
        item_set = set(self.all_data[self._ITEM])
        for i,u in enumerate(user_set):
            map_user[u] = i
        for i,iid in enumerate(item_set):
            map_item[iid] = i
        self.all_data[self._USER] = self.all_data[self._USER].map(map_user)
        self.all_data[self._ITEM] = self.all_data[self._ITEM].map(map_item)
  
    def _split_train_test(self,test_ratio=0.2):
        """
        input: data frame
        output: train_data, test_data
        """
        print("splitting train test...")
        train_data = []
        test_data = []
        # 根据时间排序分割数据集
        sort_key = None
        if self._TIME in self.all_data.columns:
            # TODO:使用datetime排序，按照字符串排序并不准确
            sort_key = self._TIME
        elif self._TIME_STAMP in self.all_data.columns:
            sort_key = self._TIME_STAMP
        
        user_grouped = self.all_data.groupby(by=['user'])
        for u_id, u_data in tqdm(user_grouped):
            if sort_key is not None:
                u_data.sort_values(by=[sort_key], inplace=True)
            else:
                # 随机排序
                u_data = u_data.sample(frac=1)
            u_data_len = len(u_data)
            train_end = int((1-test_ratio)*u_data_len)
            train_data.append(u_data.iloc[:train_end])
            test_data.append(u_data.iloc[train_end:])
        train_df = pd.concat(train_data,ignore_index=True)
        test_df = pd.concat(test_data,ignore_index=True)
        print("split train test done!")
        return train_df, test_df

    def execute(self,user_min,item_min,test_ratio):
        self.all_data = self.raw_df.copy()
        # 根据task_type 处理带有rating数据
        if 'rating' in self.all_data.columns and self.config['task_type'] == 'ranking':
            exp2imp_threshold = self.config['raw_data']['exp2imp_threshold']
            self.all_data = self.all_data[self.all_data[self._RATING]>=exp2imp_threshold]
            self.all_data = self.all_data.drop('rating', axis=1)

        self._filter_data(user_min,item_min)
        self._map_id()
        train_df,test_df = self._split_train_test(test_ratio)
        return train_df,test_df
