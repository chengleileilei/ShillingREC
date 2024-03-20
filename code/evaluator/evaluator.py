import torch

from evaluator.register import metrics_dict

class Evaluator(object):
    def __init__(self,config):
        self.task_type = config["task_type"]
        self.metric_class = {}
        self.metrics = [metric.lower() for metric in config["trainer"]["metrics"]]

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](config)
        print("metric_class: ",self.metric_class)

        self.rank_data = None
        self.rating_data = None
        # self.attack_rank_data = None


    def evaluate(self):
        # print("eval done, evaluator.rank_data.shapr" , self.rank_data.shape)
        # print("example 10: ",self.rank_data[:10,:])
        if self.task_type == "rating":
            eval_data = self.rating_data
        elif self.task_type == "ranking":
            eval_data = self.rank_data
        res = {}

        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(eval_data)
            res.update(metric_val)
        return res
    
    # def attack_evaluate(self):

    #     eval_data = self.attack_rank_data
    #     res = {}
    #     for metric in self.metrics:
    #         metric_val = self.metric_class[metric].calculate_metric(eval_data)
    #         res.update(metric_val)
    #     return res

    def update_rank_data(self,value):
        """
        value: [batch_size,rank_size+1]
        存储每个test_batch的预测矩阵，用于最终计算指标
        """
        if self.rank_data is None:
            self.rank_data = value.cpu().clone().detach()
        else:
            self.rank_data = torch.cat([self.rank_data,value.cpu().clone().detach()],dim=0)

    def update_attack_rank_data(self,value):
        """
        value: [batch_size,rank_size+1]
        存储每个test_batch的预测矩阵，用于最终计算指标
        """
        if self.attack_rank_data is None:
            self.attack_rank_data = value.cpu().clone().detach()
        else:
            self.attack_rank_data = torch.cat([self.attack_rank_data,value.cpu().clone().detach()],dim=0)
    
    def update_rating_data(self,trues,preds):
        """
        trues: [batch_size]
        preds: [batch_size]
        存储每个test_batch的预测矩阵，用于最终计算指标
        """
        if self.rating_data is None:
            self.rating_data = torch.stack([trues,preds],dim=1).cpu().clone().detach()
        else:
            self.rating_data = torch.cat([self.rating_data,torch.stack([trues,preds],dim=1).cpu().clone().detach()],dim=0)

    


    
        