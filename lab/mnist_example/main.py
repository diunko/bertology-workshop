from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING
from typing import Optional
import torch

from pytorch_lightning import Trainer, seed_everything, LightningModule

from .trainer import TrainerConfig
from .model import MNISTModel

import inspect

@dataclass
class RunConfig:
    seed: Optional[int] = None
    do_train: bool = False
    do_eval: bool = False
    do_debug: bool = False

@dataclass
class Config:
    run: RunConfig = RunConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: MNISTModel.Config = MNISTModel.Config()


def main(config: Config):
    model = MNISTModel(config.model)
    trainer = Trainer(**config.trainer)

    if config.run.do_train:
        print('training...')
        trainer.fit(model)

    if config.run.do_eval:
        print('evaluating...')

    if config.run.do_debug:
        print('debug actions...')
        # sig = inspect.signature(MNISTModel)
        # print('signature', sig)

        trainer.save_checkpoint('test123.pt')

        chk_pt = torch.load('test123.pt')

        model1 = MNISTModel.load_from_checkpoint('test123.pt')

        print('model loaded:', model1)


def cli():
    schema: Config = OmegaConf.structured(Config)
    config_yaml: Config = OmegaConf.load('lab/mnist_example/conf/main.yaml')
    config_args: Config = OmegaConf.from_cli()

    config: Config = OmegaConf.merge(schema, config_yaml, config_args)

    seed_everything(seed=config.run.seed)

    print(config.pretty())

    main(config)

if __name__ == '__main__':
    cli()