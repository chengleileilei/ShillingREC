import os
import torch
from data import RatingTaskDataLoader,RankTaskDataLoader,RankTestDataLoader,RatingTestDataLoader,PairwiseSamplerV2
import numpy as np
from evaluator import Evaluator
from rectool.utils import early_stopping
from rectool import Logger
from tqdm import tqdm

class Trainer(object):
    def __init__(self, model, config):
        self.config = config
        self.device = config['device']

        self.task_type = config["task_type"]
        self.model = model

        self.train_batch_size = self.config['trainer']["train_batch_size"]
        self.epochs = self.config['trainer']["epochs"]
        self.learning_rate = self.config['trainer']["learning_rate"]
        self.eval_batch_size = self.config['trainer']["eval_batch_size"]

        self.stopping_step = self.config['trainer']["stopping_step"]
        self.stopping_metric = self.config['trainer']["stopping_metric"].lower()
        self.stopping_bigger = self.config['trainer']["stopping_bigger"]
        self.best_score = -np.inf if self.stopping_bigger else np.inf
        self.cur_step = 0
        self.best_info = ''

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)

        self.evaluator = Evaluator(config)
        self.attack_evaluator = Evaluator(config)

        self.logger = Logger(config)
        self.logger.info("p_id:{}".format(os.getpid()))
        self.logger.info("config:{}".format(config))
        self.logger.info(self.model.dataset.statistic_info())




    def fit(self):
        if self.task_type == "rating":
            self._fit_rating_task()
        elif self.task_type == "ranking":
            self.topk = self.config['trainer']["topk"]
            self._fit_rank_task()


    def _fit_rating_task(self):
        print("getting data_iter...")
        data_iter = RatingTaskDataLoader(self.model.dataset,self.train_batch_size)
        print("getting data_iter done!")

        for epoch_idx in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for bat_users,bat_items,bat_ratings in data_iter:
                bat_users = torch.LongTensor(bat_users).to(self.device)
                bat_items = torch.LongTensor(bat_items).to(self.device)
                bat_ratings = torch.FloatTensor(bat_ratings).to(self.device)

                self.optimizer.zero_grad()
                loss = self.model.calculate_rating_loss(bat_users,bat_items,bat_ratings)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            loss_info = "Epoch{}: loss:{}".format(epoch_idx,round(epoch_loss,6))
            # self.logger.info("Epoch{}: loss:{}".format(epoch_idx,epoch_loss))

            eval_res = self._rating_evaluate()
            # print(eval_res)
            self.logger.info(loss_info + str(eval_res))
            current_score = eval_res[self.stopping_metric]
            # early stopping
            self.best_score, self.cur_step, stop_flag,self.best_info= early_stopping(
                current_score,
                self.best_score,
                self.cur_step,
                self.best_info,
                cur_info = str(eval_res),
                max_step = self.stopping_step,
                bigger=self.stopping_bigger)
            
            if stop_flag:
                stop_output = "Finished training, best result in \nepoch%d: %s" % (
                    epoch_idx - self.cur_step,
                    self.best_info
                )
                self.logger.info(stop_output)
                print(stop_output)
                # if verbose:
                #     self.logger.info(stop_output)
                break
    
    def _rating_evaluate(self):
        self.model.eval()
        test_iter = RatingTestDataLoader(self.model.dataset,self.eval_batch_size)
        for bat_test_user,bat_test_item,bat_test_rating in test_iter:
            bat_test_user = torch.LongTensor(bat_test_user).to(self.device)
            bat_test_item = torch.LongTensor(bat_test_item).to(self.device)
            trues = torch.FloatTensor(bat_test_rating).to(self.device)

            preds = self.model.predict(bat_test_user,bat_test_item)

            self.evaluator.update_rating_data(trues,preds)
        eval_res = self.evaluator.evaluate()
        self.evaluator.rating_data = None
        return eval_res



    def _fit_rank_task(self):
        print("getting data_iter...")
        data_iter = PairwiseSamplerV2(self.model.dataset,batch_size = self.train_batch_size)
        print("getting data_iter done!")
        for epoch_idx in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for bat_users,bat_pos_items,bat_neg_items in tqdm(data_iter,desc='rank train'):
                bat_users = torch.LongTensor(bat_users).to(self.device)
                bat_pos_items = torch.LongTensor(bat_pos_items).to(self.device)
                bat_neg_items = torch.LongTensor(bat_neg_items).to(self.device)
                self.optimizer.zero_grad()
                loss = self.model.calculate_rank_loss(bat_users,bat_pos_items,bat_neg_items)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            rec_eval_res,attacl_eval_res = self._rank_evaluate()
            self.logger.info("Epoch{}: loss:{}".format(epoch_idx,round(epoch_loss,6)))
            self.logger.info("Rec metric:{}".format(rec_eval_res))
            self.logger.info("Attack metric:{}".format(attacl_eval_res))


            # early stopping
            current_score = rec_eval_res[self.stopping_metric]
            self.best_score, self.cur_step, stop_flag,self.best_info= early_stopping(
                current_score,
                self.best_score,
                self.cur_step,
                self.best_info,
                cur_info = str((rec_eval_res,attacl_eval_res)),
                max_step = self.stopping_step,
                bigger=self.stopping_bigger)
            
            if stop_flag:
                stop_output = "Finished training, best result in \nepoch%d: %s" % (
                    epoch_idx - self.cur_step,
                    self.best_info
                )
                self.logger.info(stop_output)
                print(stop_output)
                # if verbose:
                #     self.logger.info(stop_output)
                break

    def _rank_evaluate(self):
        self.model.eval()
        test_iter = RankTestDataLoader(self.model.dataset, batch_size = self.eval_batch_size)
        for bat_test_user,history_index, positive_u, positive_i in test_iter:
            bat_test_user = torch.LongTensor(bat_test_user).to(self.device)

            scores_tensor = self.model.full_sort_predict(bat_test_user)
            # 训练集中的正样本设置为-inf
            scores_tensor[history_index] = -np.inf

            _, topk_idx = torch.topk(
                scores_tensor, max(self.topk), dim=-1
            )  # n_users x k
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1  # positive_u 为用户索引，positive_i 为物品索引，分别对应user列和item列

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.evaluator.update_rank_data(result)
            
            
            # print("rank task eval res:",result)
        eval_res = self.evaluator.evaluate()
        self.evaluator.rank_data = None



        for bat_test_user,history_index, _, _ in test_iter:
            bat_test_user = torch.LongTensor(bat_test_user).to(self.device)

            scores_tensor = self.model.full_sort_predict(bat_test_user)
            # 训练集中的正样本设置为-inf
            scores_tensor[history_index] = -np.inf

            _, topk_idx = torch.topk(
                scores_tensor, max(self.topk), dim=-1
            )  # n_users x k
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)

            positive_i = []
            positive_u = []
            for user in bat_test_user:
                positive_i.append(self.config['target_id_list'])
            positive_u = [np.full_like(pos_iid, i) for i, pos_iid in enumerate(positive_i)]
            positive_i = np.concatenate(positive_i)
            positive_u = np.concatenate(positive_u)


            pos_matrix[positive_u, positive_i] = 1  # positive_u 为用户索引，positive_i 为物品索引，分别对应user列和item列

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.attack_evaluator.update_rank_data(result)
        attacl_eval_res = self.attack_evaluator.evaluate()
        self.attack_evaluator.rank_data = None




        return eval_res,attacl_eval_res
        

        
        


            # attack_positive_i = np.array(self.config['target_id_list']*len(bat_test_user))
            # attack_positive_u = np.array([np.full_like(pos_iid, i) for i, pos_iid in enumerate(attack_positive_i)])





