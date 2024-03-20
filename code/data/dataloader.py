__all__ = ['RatingTaskDataLoader', 'RankTaskDataLoader',"RankTestDataLoader", "RatingTestDataLoader","PairwiseSamplerV2"]
from rectool import DataIterator
import numpy as np
import random
from collections.abc import Iterable
from collections import OrderedDict, defaultdict
from tqdm import tqdm



class Sampler(object):
    """Base class for all sampler to sample negative items.
    """

    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
    

def _generate_positive_items(user_pos_dict):
    if not user_pos_dict:
        raise ValueError("'user_pos_dict' cannot be empty.")

    users_list, items_list = [], []
    user_n_pos = dict()

    for user, items in user_pos_dict.items():
        items_list.append(items)
        users_list.append(np.full_like(items, user))
        user_n_pos[user] = len(items)
    users_arr = np.concatenate(users_list)
    items_arr = np.concatenate(items_list)
    return user_n_pos, users_arr, items_arr


def _sampling_negative_items(user_n_pos, num_neg, num_items, user_pos_dict):
    if num_neg <= 0:
        raise ValueError("'neg_num' must be a positive integer.")

    neg_items_list = []
    for user, n_pos in user_n_pos.items():
        neg_items = randint_choice(num_items, size=n_pos*num_neg, exclusion=user_pos_dict[user])
        if num_neg == 1:
            neg_items = neg_items if isinstance(neg_items, Iterable) else [neg_items]
            neg_items_list.append(neg_items)
        else:
            neg_items = np.reshape(neg_items, newshape=[n_pos, num_neg])
            neg_items_list.append(neg_items)

    return np.concatenate(neg_items_list)

def randint_choice(high, size=None, replace=True, exclusion=None):
    """从[0,high)中随机选择size个数，可以排除exclusion中的数
    excusion: list
    """
    if exclusion is None:
        return np.random.choice(high, size=size, replace=replace)
    else:
        # available_indices = np.array(list(set(range(high)) - set(exclusion)))
        # res = np.random.choice(available_indices,size=size,replace=replace)
        res = np.random.choice(high, size=size, replace=replace)
        res = set(res) - set(exclusion)
        while (len(res) != size):
            res.add(np.random.choice(high, size=1, replace=False)[0])
        return res
        # return np.random.choice([i for i in range(high) if i not in exclusion], size=size, replace=replace)

class RatingTaskDataLoader(object):
    """Data loader for rating prediction task.
    """

    def __init__(self, dataset,  batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):

        data_iter = DataIterator(self.dataset.train_user_col, self.dataset.train_item_col,self.dataset.train_rating_col, batch_size=self.batch_size, shuffle=self.shuffle)
        for bat_users,bat_items,bat_ratings in data_iter:
            yield np.asarray(bat_users),np.asarray(bat_items),np.asarray(bat_ratings)
    
    def __len__(self):
        return len(self.dataset.train_user_col) // self.batch_size

class RatingTestDataLoader(object):

    def __init__(self, dataset,  batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):

        data_iter = DataIterator(self.dataset.test_user_col, self.dataset.test_item_col,self.dataset.test_rating_col, batch_size=self.batch_size, shuffle=self.shuffle)
        for bat_users,bat_items,bat_ratings in data_iter:
            yield np.asarray(bat_users),np.asarray(bat_items),np.asarray(bat_ratings)
    
    def __len__(self):
        return len(self.dataset.train_user_col) // self.batch_size


class RankTaskDataLoader(object):
    """Data loader for ranking task.
    """

    def __init__(self, dataset, num_neg=1, batch_size=512, shuffle=False):
        """Initializes a new `RankTaskDataLoader` instance.

        Args:
            user_pos_dict (dict): A dict contains user's positive items.
            num_neg (int): Number of negative items to sample for each user.
            num_items (int): Number of items in the dataset.
            batch_size (int): Size of mini-batch.
            shuffle (bool): If `True`, the sampler will shuffle the data source.
        """
        user_pos_dict = dataset.train_dic
        self.user_n_pos, self.users_arr, self.items_arr = _generate_positive_items(user_pos_dict)
        self.neg_items_arr = _sampling_negative_items(self.user_n_pos, num_neg, dataset.n_items, user_pos_dict)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            perm = np.random.permutation(len(self.users_arr))
            users_arr = self.users_arr[perm]
            items_arr = self.items_arr[perm]
            neg_items_arr = self.neg_items_arr[perm]
        else:
            users_arr = self.users_arr
            items_arr = self.items_arr
            neg_items_arr = self.neg_items_arr

        for i in range(0, len(users_arr), self.batch_size):
            yield users_arr[i:i+self.batch_size], items_arr[i:i+self.batch_size], neg_items_arr[i:i+self.batch_size]

    def __len__(self):
        return len(self.users_arr) // self.batch_size
    
def _pairwise_sampling_v2(user_pos_dict, num_samples, num_item):
    if not isinstance(user_pos_dict, dict):
        raise TypeError("'user_pos_dict' must be a dict.")

    if not user_pos_dict:
        raise ValueError("'user_pos_dict' cannot be empty.")

    user_arr = np.array(list(user_pos_dict.keys()), dtype=np.int32)
    user_idx = randint_choice(len(user_arr), size=num_samples, replace=True)
    users_list = user_arr[user_idx] # [u1,u2,u1,u4,u1,u5,u9,,,] 

    # count the number of each user, i.e., the numbers of positive and negative items for each user
    user_pos_len = defaultdict(int)
    for u in users_list:
        user_pos_len[u] += 1
    # print("user_pos_len:",user_pos_len)

    user_pos_sample = dict()
    user_neg_sample = dict()
    for user, pos_len in tqdm(user_pos_len.items(),desc='_pairwise_sampling_v2'):

        pos_items = user_pos_dict[user]
        pos_items = np.array(pos_items)
        pos_idx = randint_choice(len(pos_items), size=pos_len, replace=True)
        pos_idx = pos_idx if isinstance(pos_idx, Iterable) else [pos_idx]

        # t = pos_items[pos_idx] 
        # t = list(t)
        user_pos_sample[user] = list(pos_items[pos_idx])

        neg_items = randint_choice(num_item, size=pos_len, replace=True, exclusion=user_pos_dict[user])

        user_neg_sample[user] = neg_items if isinstance(neg_items, Iterable) else [neg_items]
        user_neg_sample[user] = list(user_neg_sample[user])

    pos_items_list = [user_pos_sample[user].pop() for user in users_list]
    neg_items_list = [user_neg_sample[user].pop() for user in users_list]

    return users_list, pos_items_list, neg_items_list


class PairwiseSamplerV2(Sampler):
    """Sampling negative items and construct pairwise training instances.

    The training instances consist of `batch_user`, `batch_pos_item` and
    `batch_neg_items`, where `batch_user` and `batch_pos_item` are lists
    of users and positive items with length `batch_size`, and `neg_items`
    does not interact with `user`.

    If `neg_num == 1`, `batch_neg_items` is also a list of negative items
    with length `batch_size`;  If `neg_num > 1`, `batch_neg_items` is an
    array like list with shape `(batch_size, neg_num)`.
    """
    def __init__(self, dataset, num_neg=1, batch_size=1024, shuffle=True, drop_last=False):
        """Initializes a new `PairwiseSampler` instance.

        Args:
            dataset (data.Dataset): An instance of `Dataset`.
            num_neg (int): How many negative items for each positive item.
                Defaults to `1`.
            batch_size (int): How many samples per batch to load.
                Defaults to `1024`.
            shuffle (bool): Whether reshuffling the samples at every epoch.
                Defaults to `True`.
            drop_last (bool): Whether dropping the last incomplete batch.
                Defaults to `False`.
        """
        super(PairwiseSamplerV2, self).__init__()
        if num_neg <= 0:
            raise ValueError("'num_neg' must be a positive integer.")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_neg = num_neg
        self.num_items = dataset.n_items
        self.user_pos_dict =  dataset.train_dic
        # self.user_pos_set_dict = dataset.train_set_dic
        # self.user_neg_dic = dataset.train_neg_set_dic
        # self.num_trainings = sum([len(item) for u, item in self.user_pos_dict.items()]) # 所有user的pos_item数量
        self.num_trainings = dataset.train_df.shape[0]
        # self.user_pos_dict = {u: np.array(item) for u, item in user_pos_dict.items()}

    def __iter__(self):
        users_list, pos_items_list, neg_items_list = \
            _pairwise_sampling_v2(self.user_pos_dict, self.num_trainings, self.num_items)

        data_iter = DataIterator(users_list, pos_items_list, neg_items_list,
                                 batch_size=self.batch_size,
                                 shuffle=self.shuffle, drop_last=self.drop_last)
        for bat_users, bat_pos_items, bat_neg_items in data_iter:
            yield np.asarray(bat_users), np.asarray(bat_pos_items), np.asarray(bat_neg_items)

    def __len__(self):
        n_sample = self.num_trainings
        if self.drop_last:
            return n_sample // self.batch_size
        else:
            return (n_sample + self.batch_size - 1) // self.batch_size


class RankTestDataLoader(object):
    """Data loader for user prediction task.
        positive_u: user列
        positive_i: item列
    """
    def __init__(self,dataset,batch_size,shuffle=False):
        self.dataset = dataset
        self.test_dic = dataset.test_dic
        self.train_dic = dataset.train_dic
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.test_users = list(self.test_dic.keys())

    def __iter__(self):
        
        data_iter = DataIterator(self.test_users,  batch_size=self.batch_size, shuffle=self.shuffle)
        for bat_users in data_iter:
            # TODO positive_u 和 history_u 需要映射为batch id
            positive_u = []
            positive_i = []
            for user in bat_users:
                positive_i.append(self.test_dic[user])
            positive_u = [np.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_i)]

            history_u = []
            history_i = []
            for user in bat_users:
                history_i.append(self.train_dic[user])
            history_u = [np.full_like(history_iid,i) for i, history_iid in enumerate(history_i)]

            history_index = (np.concatenate(history_u),np.concatenate(history_i))
            yield np.asarray(bat_users),history_index, np.concatenate(positive_u), np.concatenate(positive_i)
    
    def __len__(self):
        return len(self.dataset.train_user_col) // self.batch_size