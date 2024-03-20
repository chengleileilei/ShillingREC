from rectool import Configurator
from attack import Attackdata
from data import Dataset
from rectool import get_rec_model,set_seed
from model import Trainer
import torch
from rectool import choose_gpu
import os

def main(config):
    c.extract_config(rec_model_path,'rec_model','rec_model_p') 
    c.extract_config(attack_model_path,'attack_model','attack_model_p')
    c.extract_config(trainer_path,'task_type','trainer')
    c.extract_config(raw_data_path,'data_name','raw_data')

    ad = Attackdata(config)

    train,test = ad.get_attacked_data(config)
    dataset = Dataset(train,test,config)

    model = get_rec_model(config['rec_model'])(config,dataset).to(config['device'])
    trainer = Trainer(model,config)
    trainer.fit()

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.dirname(__file__))
    run_path = os.path.join(root_dir,'./configs/antecedent.yaml')
    rec_model_path = os.path.join(root_dir,'./configs/rec_model.yaml')
    attack_model_path = os.path.join(root_dir,'./configs/attack_model.yaml')
    trainer_path = os.path.join(root_dir,'./configs/trainer.yaml')
    raw_data_path = os.path.join(root_dir,'./configs/raw_data.yaml')

    c = Configurator()
    c.add_config(run_path)
    device = (
            torch.device("cpu")
            if c['device']=='cpu' or not torch.cuda.is_available()
            else torch.device("cuda:"+str(choose_gpu()))
            )
    print(torch.cuda.is_available())
    print(device)
    c['device'] = device
    set_seed(c['seed'])


    # attacker_nums = [100,300,500,700,900]
    # models = ["MF", "LightGCN",  "SGL"]
    # for model in models:
    #     for attacker_num in attacker_nums:
    #         config['attacker_num'] = attacker_num
    #         config['model_name'] = model
    #         main(config,c)

    main(c)


    # attack_model = ["none", "RandomAttack", 'AverageAttack', 'LoveHate', 'AUSH']
    # for model in attack_model:
    #     c['attack_model'] = model
    #     main(c)
