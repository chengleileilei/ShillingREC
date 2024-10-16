from rectool import Configurator
from attack import Attackdata
from data import Dataset
from rectool import get_rec_model,set_seed
from model import Trainer


def generalRunner(config):
    ad = Attackdata(config)

    train,test = ad.get_attacked_data(config)
    dataset = Dataset(train,test,config)

    model = get_rec_model(config['rec_model'])(config,dataset).to(config['device'])
    trainer = Trainer(model,config)
    trainer.fit()