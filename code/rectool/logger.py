import os
import sys
import logging
from .utils import fileRename

class Logger(object):
    """Logging object.
    """
    def __init__(self, config, log_level=logging.INFO):
        # config = config["run"]
        self._logger = logging.getLogger()
        self._logger.setLevel(log_level)
        self._logger.handlers = []

        # formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        formatter = logging.Formatter("%(levelname)s-%(message)s")
        
        base_dir = os.path.abspath(os.path.dirname(__file__))
        log_dir = os.path.join(base_dir,'../',"./logs")
        data_name = config["data_name"]

        # preprocess config
        user_min = config["user_min"]
        item_min = config["item_min"]
        test_ratio = config["test_ratio"]
        preprocess_name = "user_min_{}_item_min_{}_test_ratio_{}_seed_{}".format(user_min, item_min, test_ratio, config["seed"])

        # attack config
        attacker_num = config["attacker_num"]
        filler_num = config["filler_num"]
        target_num = config["target_num"]
        attack_set = "attack_num_{}_filler_num_{}_target_num_{}".format(attacker_num, filler_num, target_num)

       
        target_strategy = "target_item_strategy_{}_{}".format(config["target_item_strategy"], config["target_num"])

        # attack_model config
        if 'attack_model_p' in config:
            attack_model_p = "_".join([config["attack_model"]]+[f"{k}_{v}" for k, v in config['attack_model_p'].items()])
        else:
            attack_model_p = "_".join([config["attack_model"]])  #config["attack_model"] 

        task_type = config["task_type"]
        rec_model = config["rec_model"]
        rec_model_p = "_".join([f"{k}_{v}" for k, v in config['rec_model_p'].items()])

        # train config
        lr = config['trainer']["learning_rate"]
        train_batch_size = config['trainer']["train_batch_size"]
        file_name = "lr_{}_batch_size_{}.log".format(lr, train_batch_size)

        if config['task_type'] == 'ranking' and 'exp2imp_threshold' in config['raw_data']:
            data_name += '_exp2imp_threshold_%s'%(config['raw_data']['exp2imp_threshold'])

        if config['attack_model'] == None or config['attack_model'] == False or config['attack_model'] == 'none':
            log_dir = os.path.join(log_dir, task_type, data_name, preprocess_name,target_strategy, attack_model_p, rec_model, rec_model_p)
        else:
            log_dir = os.path.join(log_dir, task_type, data_name, preprocess_name,target_strategy, attack_set, attack_model_p, rec_model, rec_model_p)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file = os.path.join(log_dir, file_name)
        log_file = fileRename(log_file)
        print("log file path:{}".format(log_file))

        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            self._logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        self._logger.addHandler(ch)

    def get_logger(self):
        return self._logger

    def set_level(self, level):
        self._logger.setLevel(level)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)

    def critical(self, msg):
        self._logger.critical(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def exception(self, msg):
        self._logger.exception(msg)