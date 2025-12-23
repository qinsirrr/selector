import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from module.lightning_selector import LightningForSelector
from utils.functions import setup_parser
from utils.dataset_seletor import DataModuleForSelector

if __name__ == '__main__':
    args = setup_parser()
    data_module = DataModuleForSelector(args)
    lightning_model =LightningForSelector(args)
    trainer = pl.Trainer(**args.selectorTrainer)
    trainer.fit(lightning_model, data_module)
    trainer.test(lightning_model, data_module)
    # trainer.validate(lightning_model, data_module)

