import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import torch
from tqdm import tqdm

_USER = "user"
_ITEM = "item"
_RATING = "rating"
_TIME = "time"
_TIME_STAMP = "time_stamp"
_LABEL = "label"

# 针对显示数据和隐式数据分别设计Dataset

class Dataset(object):
    def __init__(self,train_df,test_df,config):
        self.task_type = config["task_type"]
        self.train_df = train_df
        self.test_df = test_df

        self.n_users = max(self.train_df[_USER].max(),self.test_df[_USER].max())+1
        self.n_items = max(self.train_df[_ITEM].max(),self.test_df[_ITEM].max())+1

        self.train_user_col = self.train_df[_USER].values
        self.train_item_col = self.train_df[_ITEM].values
        self.test_user_col = self.test_df[_USER].values
        self.test_item_col = self.test_df[_ITEM].values

        if self.task_type == "ranking":
            self.train_dic = self.df2dic(self.train_df)
            print('train_dic done!')
            self.test_dic = self.df2dic(self.test_df)
            # self.train_pos_set_dic = {k:set(v) for k,v in self.train_dic.items()}
            # print('train_pos_set_dic done!')
            # print('get train_neg_set_dic....')
            # all_item_set = set(range(self.n_items)) 
            # self.train_neg_set_dic = {k:(all_item_set-v) for k,v in tqdm(self.  .items())}
            # self.train_neg_set_dic = {}
            # for k,v in tqdm(self.train_pos_set_dic.items()):
            #     self.train_neg_set_dic[k] = all_item_set-v
            # print('train_neg_set_dic done!')
            # self.test_set_dic = {k:set(v) for k,v in self.test_dic.items()}

        if self.task_type == 'rating':
            self.test_rating_col = self.test_df[_RATING].values
            self.train_rating_col = self.train_df[_RATING].values
            self.train_rating_min = self.train_df[_RATING].min()
            self.train_rating_max = self.train_df[_RATING].max()

    def df2dic(self,df):
        res_dic = {}
        for user, item in zip(df[_USER],df[_ITEM]):
            if user not in res_dic:
                res_dic[user] = []
            res_dic[user].append(item)
        return res_dic

    def df2setdic(self,df):
        res_set = {}
        for user, item in zip(df[_USER],df[_ITEM]):
            if user not in res_set:
                res_set[user] = set()
            res_set[user].add(item)
        return res_set
        

    def get_rec_train_test(self,user_min,item_min,test_ratio):
        pass

    def get_user_features(self):
        pass
    def get_dete_train_test(self,test_ratio):
        pass

    def inter_matrix(self,form="coo",value_field=None):
        src = self.train_user_col
        tgt = self.train_item_col
        if value_field is None:
            data = np.ones(len(self.train_df))  # 
        print("src.shape:{},tgt.shape:{},data.shape:{}".format(src.shape,tgt.shape,data.shape))
        print("src:{},tgt:{},data:{}".format(src,tgt,data))
        shape = (self.n_users,self.n_items)
        mat = coo_matrix(
            (data, (src, tgt)), shape=shape)

        if form == "coo":
            return mat
        elif form == "csr":
            return mat.tocsr()
        # else:
        #     raise NotImplementedError(
        #         f"Sparse matrix format [{form}] has not been implemented."
        #     )

    def statistic_info(self):
        train_info = self.get_df_info(self.train_df)
        test_info = self.get_df_info(self.test_df)
        common_info = self.get_common_info(self.train_df,self.test_df)
        train_info = "Train data info:\n" + "\n".join(train_info)
        test_info = "Test data info:\n" + "\n".join(test_info)
        common_info = "Common info:\n" + "\n".join(common_info)
        res_str = "\n".join([train_info,test_info,common_info])
        res_str = "Dataset info\n" + res_str + "\n"
        return res_str

    def get_df_info(self,df):
        res = []

        user_num = len(df[_USER].unique())
        item_num = len(df[_ITEM].unique())
        df_len = df.shape[0]
        if _RATING in df.columns:
            rating_info = self.value_info(df[_RATING])
            res.append("rating_info:{}".format(rating_info))
        if _LABEL in df.columns:
            label_info = self.value_info(df[_LABEL])
            res.append("label_info:{}".format(label_info))

        res.append("user_num:{}".format(user_num))
        res.append("item_num:{}".format(item_num))
        res.append("df_len:{}".format(df_len))
        res.append("average of action per user:{:.2f}".format(df_len/user_num))
        res.append("average of action per item:{:.2f}".format(df_len/item_num))
        res.append("sparsity  {:.2f}".format(df_len/(user_num*item_num)*100))
        return res

    def value_info(self,df_col):
        """
        统计df_col中每个值出现的次数和比例,并根据key排序
        output: {value1:count1, value2:count2, ...}
        """
        value_count = df_col.value_counts()
        value_count = value_count.sort_index()
        value_count = value_count.to_dict()
        value_count = {k:v for k,v in value_count.items()}
        return value_count

    def get_common_info(self,train_df,test_df):
        """
        统计train_df和test_df中共有的user和item数量
        """
        common_user = set(train_df[_USER].unique()) & set(test_df[_USER].unique())
        common_item = set(train_df[_ITEM].unique()) & set(test_df[_ITEM].unique())
        res = []
        res.append("common_user_num:{}".format(len(common_user)))
        res.append("common_item_num:{}".format(len(common_item)))
        return res
    
    def _history_matrix(self, row, value_field=None, max_history_len=None):
        """Get dense matrix describe user/item's history interaction records.

        ``history_matrix[i]`` represents ``i``'s history interacted item_id.

        ``history_value[i]`` represents ``i``'s history interaction records' values.
            ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            row (str): ``user`` or ``item``.
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.
            max_history_len (int): The maximum number of history interaction records.
                Defaults to ``None``.

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        # self._check_field("uid_field", "iid_field")

        # inter_feat = copy.deepcopy(self.inter_feat)
        # inter_feat.shuffle()
        user_ids, item_ids = (
            self.train_user_col,
            self.train_item_col,
        )

        # if value_field is None:
        #     values = np.ones(len(user_ids))
        # else:
        #     if value_field not in inter_feat:
        #         raise ValueError(
        #             f"Value_field [{value_field}] should be one of `inter_feat`'s features."
        #         )
        #     values = inter_feat[value_field].numpy()

        if self.task_type == 'rating':
            values = self.train_rating_col
        else:
            values = np.ones(len(user_ids))



        if row == "user":
            row_num, max_col_num = self.n_users, self.n_items
            row_ids, col_ids = user_ids, item_ids
        else:
            row_num, max_col_num = self.n_items, self.n_users
            row_ids, col_ids = item_ids, user_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1

        max_inter_num = np.max(history_len)
        if max_history_len is not None:
            col_num = min(max_history_len, max_inter_num)
        else:
            col_num = max_inter_num

        # if col_num > max_col_num * 0.2:
        #     self.logger.warning(
        #         f"Max value of {row}'s history interaction records has reached "
        #         f"{col_num / max_col_num * 100}% of the total."
        #     )

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_value = np.zeros((row_num, col_num))
        history_len[:] = 0
        for row_id, value, col_id in zip(row_ids, values, col_ids):
            if history_len[row_id] >= col_num:
                continue
            history_matrix[row_id, history_len[row_id]] = col_id
            history_value[row_id, history_len[row_id]] = value
            history_len[row_id] += 1

        return (
            torch.LongTensor(history_matrix),
            torch.FloatTensor(history_value),
            torch.LongTensor(history_len),
        )

    def history_item_matrix(self, value_field=None, max_history_len=None):
        """Get dense matrix describe user's history interaction records.

        ``history_matrix[i]`` represents user ``i``'s history interacted item_id.

        ``history_value[i]`` represents user ``i``'s history interaction records' values,
        ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of user ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.

            max_history_len (int): The maximum number of user's history interaction records.
                Defaults to ``None``.

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        return self._history_matrix(
            row="user", value_field=value_field, max_history_len=max_history_len
        )
    


class dfDataset(object):
    def __init__(self,df,config):
        self.config = config
        self.df = df
        self.n_users = self.df[_USER].max()+1
        self.n_items = self.df[_ITEM].max()+1

        self.user_col = self.df[_USER].values
        self.item_col = self.df[_ITEM].values
        if config['task_type'] == 'rating':
            self.rating_col = self.df[_RATING].values
            self.rating_min = self.df[_RATING].min()
            self.rating_max = self.df[_RATING].max()

        self.user_item_dic = self.df2dic(df)
        self.mat = None


    def df2dic(self,df):
        res_dic = {}
        for user, item in zip(df[_USER],df[_ITEM]):
            if user not in res_dic:
                res_dic[user] = []
            res_dic[user].append(item)
        return res_dic
    
    def to_matrix(self):
        if self.config['task_type'] == 'rating':
            values = self.rating_col
        else:
            values = np.ones(len(self.df))

        row, col = self.user_col, self.item_col
        # print("row type:",type(row),row.dtype)
        # print("col type:",type(col),col.dtype)
        # print("rating type:",type(rating),rating.dtype)
        row = row.astype("int64")
        col = col.astype("int64")
        values = values.astype("float32")
        matrix = csr_matrix((values, (row, col)), shape=(self.n_users, self.n_items)).toarray()
        self.mat = matrix
        return matrix

    def generate_batch(self, user_filter):
        # user_filter = config.get("user_filter", None)
        # if self.mode() == 'train':
        batch_size = self.config['trainer']["train_batch_size"]
        if user_filter is not None:
            available_idx = user_filter(train_mat=self.mat)
        else:
            available_idx = list(range(len(self.mat)))
        available_idx = np.random.permutation(available_idx)
        total_batch = (len(available_idx) + batch_size - 1) // batch_size
        for b in range(total_batch):
            batch_set_idx = available_idx[b * batch_size : (b + 1) * batch_size]
            real_profiles = self.mat[batch_set_idx, :].astype('float')

            yield {
                "users": torch.tensor(batch_set_idx, dtype=torch.int64).to(
                    self.config['device']
                ),
                "users_mat": torch.tensor(real_profiles, dtype=torch.float32).to(
                    self.config['device']
                ),
            }








