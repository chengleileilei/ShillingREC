import torch
import time
from model import sgdl_training
from model.rec.sgdl import LightGCN, LTW
import model
import pickle
import rectool
import data
# import parse
# from parse import config, log_file
from rectool import Logger
from prettytable import PrettyTable
from attack import Attackdata
import numpy as np
import os

def SGDLRunner(config):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    tmp_path = os.path.join(base_dir,'../../',"./tmp_data")
    
    data_name = config["data_name"]
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
    rec_model_p = "_".join([f"{k}_{v}" for k, v in config["rec_model_p"].items()])

    if rec_model == 'SGDL':
        rec_model_p = ''

    # train config
    lr = config['trainer']["learning_rate"]
    train_batch_size = config['trainer']["train_batch_size"]
    file_name = "lr_{}_batch_size_{}.log".format(lr, train_batch_size)

    if config['task_type'] == 'ranking' and 'exp2imp_threshold' in config['raw_data']:
        data_name += '_exp2imp_threshold_%s'%(config['raw_data']['exp2imp_threshold'])

    if config['attack_model'] == None or config['attack_model'] == False or config['attack_model'] == 'none':
        tmp_path = os.path.join(tmp_path, task_type, data_name, preprocess_name,target_strategy, attack_model_p, rec_model, rec_model_p)
    else:
        tmp_path = os.path.join(tmp_path, task_type, data_name, preprocess_name,target_strategy, attack_set, attack_model_p, rec_model, rec_model_p)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    config["path"] = tmp_path





    ad = Attackdata(config)
    train_df,test_df = ad.get_attacked_data(config)
    # train_df 保留user item label 列，并将label重命名未noisy，类型为dtype={0: np.int32, 1: np.int32, 2: np.int32}
    train_df = train_df[['user', 'item', 'label']]
    train_df.columns = ['user', 'item', 'noisy']
    train_df = train_df.astype({'user': np.int32, 'item': np.int32, 'noisy': np.int32})
    # test_df 保留user item 列
    test_df = test_df[['user', 'item']]
    test_df = test_df.astype({'user': np.int32, 'item': np.int32})

    print(train_df.head())
    print(test_df.head())

    logger = Logger(config)









    rectool.set_seed(config['seed'])
    mem_manager = data.MemLoader(config,train_df)
    train_dataset = data.Loader(config,train_df,test_df)

    Recmodel = LightGCN(config, train_dataset)
    Recmodel = Recmodel.to(config["device"])
    ltw = LTW(config["rec_model_p"]["input"], config["rec_model_p"]["hidden1"], config["rec_model_p"]["output"]).cuda()
    logger.info(str(config))
    results = []
    # attack_results = []
    config["rec_model_p"]["lr"] /= 5
    opt = torch.optim.Adam(Recmodel.params(), lr=config["rec_model_p"]["lr"])

    # ========== Phase I: Memorization ========== #
    for epoch in range(config["rec_model_p"]["epochs"]):
        time_train = time.time()
        output_information = sgdl_training.memorization_train(config, train_dataset, Recmodel, opt)
        train_log = PrettyTable()
        train_log.field_names = ['Epoch', 'Loss', 'Time', 'Estimated Clean Ratio', 'Memory ratio']

        clean_ratio = sgdl_training.estimate_noise(config, mem_manager, Recmodel)
        mem_ratio = sgdl_training.memorization_test(config, mem_manager, Recmodel)
        train_log.add_row(
            [f'{epoch + 1}/{config["rec_model_p"]["epochs"]}', output_information, f'{(time.time() - time_train):.3f}',
            f'{clean_ratio:.5f}', f'{mem_ratio:.5f}']
        )
        logger.info(str(train_log))

        # memorization point
        if mem_ratio >= clean_ratio:
            logger.info(f'==================Memorization Point==================')
            break

    trans_epoch = epoch
    clean_dataset = data.CleanLoader(config)
    config["rec_model_p"]["lr"] *= 5
    best_epoch = epoch

    # ========== Phase II: Self-Guided Learning ========== #
    for epoch in range(trans_epoch, config["rec_model_p"]["epochs"]):
        if epoch % config["rec_model_p"]["eval_freq"] == 0:
            logger.info(f'======================Validation======================')
            valid_log = PrettyTable()
            valid_log.field_names = ['Precision', 'Recall', 'NDCG', 'HR', 'Current Best Epoch']
            valid_result = sgdl_training.test(config, logger, train_dataset, Recmodel, valid=True, multicore=config["rec_model_p"]["multicore"])
            results.append(valid_result) # rec eval
            valid_result_attack = sgdl_training.test_attack(config, logger, train_dataset, Recmodel, valid=True, multicore=config["rec_model_p"]["multicore"])
            # attack_results.append(valid_result_attack) # attack eval
            pkl_path = os.path.join(config["path"], 'results_{}_{}.pkl'.format(
                    config["data_name"],
                    config["rec_model_p"]["lr"],
                    config["rec_model_p"]["meta_lr"]
            ))
            with open(pkl_path, 'wb') as f:
                pickle.dump(results, f)
            is_stop, is_save = rectool.EarlyStop(results, config["rec_model_p"]["stop_step"])

            # save current best model
            if is_save:
                best_epoch = epoch
                save_path = os.path.join(config["path"], 'model_{}_{}_{}_{}_{}_schedule_{}.pth'.format(
                    config["rec_model_p"]["lr"],
                    config["rec_model_p"]["meta_lr"],
                    config["rec_model_p"]["model"],
                    config["rec_model_p"]["schedule_type"],
                    config["rec_model_p"]["tau"],
                    config["rec_model_p"]["schedule_lr"]
                ))
                torch.save(Recmodel.state_dict(), save_path)
            valid_log.add_row(
                [valid_result['precision'][0], valid_result['recall'][0], valid_result['ndcg'][0], valid_result['hit'][0], best_epoch]
            )
            valid_log.add_row(
                [valid_result_attack['precision'][0], valid_result_attack['recall'][0], valid_result_attack['ndcg'][0], valid_result_attack['hit'][0], best_epoch]
            )
            logger.info(str(valid_log))
            if is_stop:
                break

        time_train = time.time()
        if config["rec_model_p"]["schedule_type"] == 'reinforce':
            output_information = sgdl_training.self_guided_train_schedule_reinforce(config, train_dataset, clean_dataset, Recmodel, ltw)
        elif config["rec_model_p"]["schedule_type"] == 'gumbel':
            output_information = sgdl_training.self_guided_train_schedule_gumbel(config, train_dataset, clean_dataset, Recmodel, ltw)
        else:
            logger.info('Invalid scheduler type !')
            exit()

        train_log = PrettyTable()
        train_log.field_names = ['Epoch', 'Train Loss', "Meta Loss", "Time"]
        train_log.add_row(
            [f'{epoch + 1}/{config["rec_model_p"]["epochs"]}', output_information[0], output_information[1], f'{(time.time()-time_train):.3f}']
        )
        logger.info(str(train_log))

    # # ========== Test ========== #
    # logger.info(f'=========================Test=========================')
    # state = torch.load('./{}/model_{}_{}_{}_{}_{}_schedule_{}.pth'.format(
    #                 config["data_name"],
    #                 config["rec_model_p"]["lr"],
    #                 config["rec_model_p"]["meta_lr"],
    #                 config["rec_model_p"]["model"],
    #                 config["rec_model_p"]["schedule_type"],
    #                 config["rec_model_p"]["tau"],
    #                 config["rec_model_p"]["schedule_lr"]
    #             ))
    # Recmodel.load_state_dict(state)
    # sgdl_training.test(config, logger,train_dataset, Recmodel, valid=False, multicore=config["rec_model_p"]["multicore"])
