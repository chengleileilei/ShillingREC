__all__ = ["get_rec_model","get_attack_model","filler_filter_mat","pick_optim","set_seed","VarDim","getPopularItemId"]

import importlib
from torch import optim
import random
import numpy as np
import torch

def get_rec_model(model_name):
    """
    根据模型名称获取模型
    in: model_name:str
    out: model:class
    """
    model_path = "model.rec"
    # print(model_path)
    model_module = importlib.import_module(model_path)
    model = getattr(model_module,model_name)
    return model

def get_attack_model(model_name):
    """
    根据模型名称获取模型
    in: model_name:str
    out: model:class
    """
    model_path = "attack.attack_model"
    # print(model_path)
    model_module = importlib.import_module(model_path)
    model = getattr(model_module,model_name)
    return model

# 优化为1个函数


def filler_filter_mat(train_mat, target_id_list=[], selected_ids=[], filler_num=0):
    mask_array = (train_mat > 0).astype('float')
    # print("selected_ids:",selected_ids,type(selected_ids))
    # print("target_id_list:",target_id_list,type(target_id_list))
    # print("+",selected_ids + target_id_list)
    mask_array[:, np.concatenate([selected_ids,target_id_list])] = 0
    available_idx = np.where(np.sum(mask_array, 1) >= filler_num)[0]
    return available_idx

def pick_optim(which):
    if which.lower() == 'adam':
        return optim.Adam
    else:
        if hasattr(optim, which):
            # _logger.info(f"load {which} from torch.optim")
            return getattr(optim, which)
        else:
            raise ValueError("optimizer not supported")

class VarDim:
    def __init__(self, max=None, min=None, comment=""):
        self.max = max or "?"
        self.min = min or "0"
        self.comment = comment

    def __repr__(self) -> str:
        return f"{self.comment}[{self.min}~{self.max}]"

    def __str__(self) -> str:
        return self.__repr__()




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def getPopularItemId(item_col, k, exclusion=[]):
    # 找到k个最热门的item，且与exclusion不重复
    popular_item_list = item_col.value_counts().index
    res = []
    for item in popular_item_list:
        if item not in exclusion:
            res.append(item)
            if len(res) == k:
                break
    return res