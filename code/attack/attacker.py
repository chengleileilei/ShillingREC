__all__ = ["Attackdata"]
import os 
import pandas as pd
import numpy as np
from tqdm import tqdm

from rectool import Preprocessor
from rectool import get_attack_model

class Attackdata(object):
    _USER = "user"
    _ITEM = "item"
    _RATING = "rating"
    _TIME = "time"
    _LABEL = "label"
    def __init__(self,config):
        self.config = config
        self.data_name = config['data_name']
        self.user_min = config['user_min']
        self.item_min = config['item_min']
        self.test_ratio = config['test_ratio']

        self.attack_model = config['attack_model']
        self.attacker_num = config['attacker_num']
        self.filler_num = config['filler_num']
        self.target_num = config['target_num']

        self.attack_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.join(self.attack_dir, "..","..")

        if self.config['task_type'] == 'rating':
            self.ready_data_path = os.path.join(self.root_dir,"ready_data",self.data_name, 'explicit')
        elif self.config['task_type'] == 'ranking':
            file_name = 'implicit'
            if 'exp2imp_threshold' in self.config['raw_data']:
                file_name = 'implicit_exp2imp_threshold_%s'%(self.config['raw_data']['exp2imp_threshold'])
            self.ready_data_path = os.path.join(self.root_dir,"ready_data",self.data_name, file_name)
        self.cleaned_path = os.path.join(self.ready_data_path,"user_min_%s_item_min_%s_test_ratio_%s"%(self.user_min,self.item_min,self.test_ratio))

        self.train_df = None
        self.test_df = None

    def get_train_test(self):
        # 根据filter读取或者生成train和test数据集        
        train_path = os.path.join(self.cleaned_path,"Train","train.csv")
        test_path = os.path.join(self.cleaned_path,"Test","test.csv")

        if (not self.config["debug"]) and os.path.exists(train_path) and os.path.exists(test_path):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            file_name = self.config['raw_data']["file_name"]
            data_path = os.path.join(self.root_dir,"rawdata",self.data_name,file_name)
            sep = self.config['raw_data']["sep"]
            header = self.config['raw_data']["header"]
            names = self.config['raw_data']["names"]
            raw_df = pd.read_csv(data_path, sep=sep, header=header, names=names)

            preprocessor = Preprocessor(self.config, raw_df)
            train_df,test_df = preprocessor.execute(self.user_min, self.item_min, self.test_ratio)

            train_dir = os.path.dirname(train_path)
            test_dir = os.path.dirname(test_path)
            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)
            train_df.to_csv(train_path,index=False)
            test_df.to_csv(test_path,index=False)
        self.train_df = train_df
        self.test_df = test_df
        return train_df,test_df

    def get_attacked_data(self,config):
        if self.train_df == None or self.test_df == None:
            self.get_train_test()

        # 读取或者生成target_item_info
        if self.config['target_item_strategy'] == 'popular':
            target_item_info_path = os.path.join(self.cleaned_path,"Train","Attacked","popular_target_item_{}.log".format(config['target_num']))
        elif self.config['target_item_strategy'] == 'random':
            target_item_info_path = os.path.join(self.cleaned_path,"Train","Attacked","popular_target_item_{}.log".format(config['target_num']))
        
        if os.path.exists(target_item_info_path):
            with open(target_item_info_path,'r') as f:
                target_id_list = eval(f.read())
        else:
            target_id_list = self.get_attack_target_id_list(self.train_df)
            if not os.path.exists(os.path.dirname(target_item_info_path)):
                os.makedirs(os.path.dirname(target_item_info_path))
            # 保存target_item_info
            with open(target_item_info_path,'w') as f:
                f.write(str(list(target_id_list)))
        config['target_id_list'] = list(target_id_list)


        if self.attack_model == None or self.attack_model == False or self.attack_model == 'none':
            return self.train_df, self.test_df
        fake_df_dir = os.path.join(self.cleaned_path,"Train","Attacked",self.attack_model,"attacker_num_%s_filler_num_%s_target_num_%s_target_item_strategy_%s"%(self.attacker_num,self.filler_num,self.target_num,self.config["target_item_strategy"]))
        fake_df_path = os.path.join(fake_df_dir,"fake.csv")

        # 读取或者生成fake_df
        if (not self.config["debug"]) and os.path.exists(fake_df_path) :
            fake_df = pd.read_csv(fake_df_path)

        else:
            attack_model = get_attack_model(self.attack_model)(self.train_df,self.config)
            fake_df = attack_model.get_fake_df()

            if not os.path.exists(fake_df_dir):
                os.makedirs(fake_df_dir)
            fake_df.to_csv(fake_df_path,index=False)

        self.train_df['label'] = 0
        fake_df['label'] = 1

        # 合并fake_df 和 train_df
        attacked_train_df = pd.concat([self.train_df,fake_df],ignore_index=True)
        # config['target_id_list'] = target_id_list
        print('Train data path  %s'% os.path.normpath(fake_df_path))
        print('Test data path  %s'% os.path.normpath(self.cleaned_path))
        return attacked_train_df, self.test_df
    
    def get_attack_target_id_list(self,train_df):
        self.target_num = self.config['target_num']
        self.item_unique = np.sort(self.train_df[self._ITEM].unique())

        if self.config['target_item_strategy'] == "popular":
            # 选择流行的item
            target_id_list = train_df[self._ITEM].value_counts().index[:self.target_num]
        elif self.config['target_item_strategy'] == "random":
            target_id_list = np.random.choice(self.item_unique,self.target_num,replace=False) 
        return target_id_list


       
if __name__ == "__main__":
    data_name = "ml-100k"
    user_min = 0
    item_min = 0
    test_ratio = 0.2
    attack_model = "random"
    attacker_num = 10
    filler_num = 10
    target_num = 2
    ad = Attackdata(data_name,user_min,item_min,test_ratio)
    attacked_train_df,test = ad.get_attacked_data(attack_model, attacker_num, filler_num, target_num)




