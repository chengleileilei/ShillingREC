# __all__ = ['RatingTaskDataLoader', 'RankTaskDataLoader',"RankTestDataLoader", "RatingTestDataLoader","PairwiseSamplerV2"]
from rectool import DataIterator
import numpy as np
import random
from collections.abc import Iterable
from collections import OrderedDict, defaultdict
from tqdm import tqdm

import torch
import pandas as pd
from torch.utils.data import Dataset as torchDataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
# import parse
import os



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






# sgdl dataloader
class MemLoader(torchDataset):
    '''
    Memorization management
    Function: generate and update memorized data
    '''
    def __init__(self, config, train_df):
        self.train_df = train_df

        # self.path = f'../data/{config['data_name']}'
        self.path = config['path']
        self.dataset = config['data_name']
        self.history_len = config["rec_model_p"]["history_len"]
        self.n_user = 0
        self.m_item = 0
        self.config = config

        print('Preparing memloader...')

        # train_file = self.path + f'/{self.dataset}.train.rating'

        # train_data = pd.read_csv(
        #     train_file,
        #     sep='\t', header=None, names=['user', 'item', 'noisy'],
        #     usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32, 2: np.int32}
        # )
        train_data = train_df
        if self.dataset == 'adressa':
            self.n_user = 212231
            self.m_item = 6596
        else:
            self.n_user = train_data['user'].max() + 1
            self.m_item = train_data['item'].max() + 1

        # record number of iteractions of each user
        self.user_pos_counts = pd.value_counts(train_data['user']).sort_index()

        self.trainUniqueUsers = np.array(range(self.n_user))
        self.trainUser = train_data['user'].values
        self.trainItem = train_data['item'].values
        self.traindataSize = len(self.trainItem)

        # memorization history matrix, 1 for memorized and 0 for non-memorized
        self.mem_dict = np.zeros((self.traindataSize, self.history_len), dtype=np.int8)
        # loop pointer that indicates current epoch, increment at the beginning of each epoch
        self.mem_dict_p = -1
        # map index from (u,i) to row position of memorization history matrix
        self.index_map = np.zeros((self.n_user, self.m_item), dtype=np.int32)
        self.index_map[:, :] = -1
        for ii in range(self.traindataSize):
            u = self.trainUser[ii]
            i = self.trainItem[ii]
            self.index_map[u][i] = ii

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self._allPos = self.getUserPosItems(list(range(self.n_user)))

    def updateMemDict(self, users, items):
        '''
        users and items: memorized pairs
        '''
        # increment pointer
        self.mem_dict_p += 1
        # loop pointer
        self.mem_dict_p %= self.history_len
        # initialize (clear) memorization record of current epoch
        self.mem_dict[:, self.mem_dict_p] = 0

        indexes = []
        for i in range(len(users)):
            index = self.index_map[users[i]][items[i]]
            if index != -1:
                indexes.append(index)
        self.mem_dict[indexes, self.mem_dict_p] = 1

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self._allPos = self.getUserPosItems(list(range(self.n_user)))

    def generate_clean_data(self):
        '''
        generate memorized data
        '''
        ismem_dict = np.sum(self.mem_dict, axis=1) >= self.history_len / 2
        mem_num = np.sum(ismem_dict)
        #print('Memory ratio:', mem_num / self.traindataSize)
        if mem_num > 0:
            indexes = np.argwhere(ismem_dict == True).reshape(1, -1)[0]
            clean_us = np.array(self.trainUser)[indexes]
            clean_is = np.array(self.trainItem)[indexes]
            clean_data = {'user': clean_us, 'item': clean_is}
            df = pd.DataFrame(clean_data)
            file_name = 'clean_data_{}_{}.txt'.format(self.config['rec_model'], self.config["rec_model_p"]["lr"])
            save_path = os.path.join(self.path, file_name)

            df.to_csv(save_path, header=False, index=False,sep='\t')
            return mem_num / self.traindataSize
        else:
            return False

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def allPos(self):
        return self._allPos

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

class Loader(torchDataset):
    def __init__(self, config, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

        self.path = config['path']
        self.dataset = config['data_name']

        # print(f'loading [{self.path}]...')

        self.split = config["rec_model_p"]["A_split"]
        self.folds = config["rec_model_p"]["A_n_fold"]
        self.n_user = 0
        self.m_item = 0
        self.config = config

        # train_file = self.path + f'/{self.dataset}.train.rating'
        # test_file = self.path + f'/{self.dataset}.test.negative'
        # valid_file = self.path + f'/{self.dataset}.valid.rating'
        trainItem, trainUser = [], []
        testUniqueUsers, testItem, testUser = [], [], []

        # loading training file
        # with open(train_file, 'r') as f:
        #     line = f.readline()
        #     while line and line != '':
        #         arr = line.split('\t')
        #         u = int(arr[0])
        #         i = int(arr[1])
        #         self.m_item = max(self.m_item, i)
        #         self.n_user = max(self.n_user, u)
        #         trainUser.append(u)
        #         trainItem.append(i)
        #         line = f.readline()
        trainUser = train_df['user'].values
        trainItem = train_df['item'].values
        self.m_item = trainItem.max()
        self.n_user = trainUser.max()
        
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainUniqueUsers = np.array(list(set(trainUser)))
        self.traindataSize = len(trainUser)

        # loading validation file
        validUser, validItem, validUniqueusers = [], [], []
        # with open(valid_file, 'r') as f:
        #     line = f.readline()
        #     while line and line != '':
        #         arr = line.split('\t')
        #         u = int(arr[0])
        #         i = int(arr[1])
        #         self.m_item = max(self.m_item, i)
        #         self.n_user = max(self.n_user, u)
        #         validUser.append(u)
        #         validItem.append(i)
        #         line = f.readline()
        validUser = test_df['user'].values
        validItem = test_df['item'].values
        self.m_item = max(self.m_item, validItem.max())
        self.n_user = max(self.n_user, validUser.max())

        self.validUser = np.array(validUser)
        self.validItem = np.array(validItem)
        self.validUniqueUsers = np.array(list(set(validUser)))
        self.validdataSize = len(self.validItem)

        # loading test file
        # with open(test_file, 'r') as f:
        #     line = f.readline()
        #     while line and line != '':
        #         arr = line.split('\t')
        #         if self.dataset == 'adressa':
        #             u = eval(arr[0])[0]
        #             i = eval(arr[0])[1]
        #         else:
        #             u = int(arr[0])
        #             i = int(arr[1])
        #         self.m_item = max(self.m_item, i)
        #         self.n_user = max(self.n_user, u)
        #         testUser.append(u)
        #         testItem.append(i)
        #         line = f.readline()

        self.m_item += 1
        self.n_user += 1
        # self.testUser = np.array(testUser)
        # self.testItem = np.array(testItem)
        # self.testUniqueUsers = np.array(list(set(testUser)))

        # self.testdataSize = len(self.testItem)

        self.Graph = None

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        # self.__testDict = self.__build_test()

        self.__validDict = self.__build_valid()

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def evalDict(self):
        return self.__evalDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.config['device']))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(f'{self.path}/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(f'{self.path}/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(self.config['device'])
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def __build_valid(self):
        valid_data = {}
        for i, item in enumerate(self.validItem):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(item)
            else:
                valid_data[user] = [item]
        return valid_data

    def __build_eval(self):
        eval_data = {}
        for i, item in enumerate(self.trainItem):
            user = self.trainUser[i]
            if eval_data.get(user):
                eval_data[user].append(item)
            else:
                eval_data[user] = [item]
        return eval_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

class CleanLoader(torchDataset):
    def __init__(self, config):
        self.path = config['path']
        self.split = config["rec_model_p"]["A_split"]
        self.folds = config["rec_model_p"]["A_n_fold"]
        self.n_user = 0
        self.m_item = 0
        self.config = config
        train_file = self.path + '/clean_data_{}_{}.txt'.format(self.config['rec_model'], self.config["rec_model_p"]["lr"])

        trainItem, trainUser = [], []

        with open(train_file, 'r') as f:
            line = f.readline()
            while line and line != '':
                arr = line.split('\t')
                u = int(arr[0])
                i = int(arr[1])
                self.m_item = max(self.m_item, i)
                self.n_user = max(self.n_user, u)
                # print(self.m_item)
                trainUser.append(u)
                trainItem.append(i)
                line = f.readline()
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)
        self.trainUniqueUsers = np.array(list(set(trainUser)))
        self.traindataSize = len(trainUser)

        self.m_item += 1
        self.n_user += 1

        self.Graph = None

        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = None
        self.__validDict = None

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems