import os
import logging

logger = logging.getLogger(__name__)

from typing import Optional
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING

import pytorch_lightning as pl

from .trainer import TrainerConfig
from .model import MRPCTransformer

import inspect

def set_seed(seed: Optional[int]=None, gpus=None):
    seed = pl.seed_everything(seed)
    if gpus:
        torch.cuda.manual_seed_all(seed)

class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))

@dataclass
class RunConfig:
    do_train: bool = False
    do_eval: bool = False
    do_debug: bool = False
    seed: Optional[int] = None
    output_dir: Optional[str] = None

@dataclass
class Config:
    run: RunConfig = RunConfig()
    trainer: TrainerConfig = TrainerConfig()
    model: MRPCTransformer.Config = MRPCTransformer.Config()

def cli():

    # get configuration
    schema: Config = OmegaConf.structured(Config)
    config_yaml: Config = OmegaConf.load(f'{os.path.dirname(__file__)}/conf/main.yaml')
    config_args: Config = OmegaConf.from_cli()

    config: Config = OmegaConf.merge(schema, config_yaml, config_args)

    if config.run.output_dir is None:
        config.run.output_dir = os.path.join("./results", f"{config.model.task}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(config.run.output_dir)

    print(config.pretty())

    # setup run
    if os.path.exists(config.run.output_dir) and os.listdir(config.run.output_dir) and config.run.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(config.run.output_dir))

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=config.run.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=5
    )

    # setup model and trainer
    set_seed(config.run.seed, config.trainer.gpus)

    model = MRPCTransformer(config.model)
    trainer = pl.Trainer(
        **config.trainer,
        **dict(
            checkpoint_callback=checkpoint_callback,
            callbacks=[LoggingCallback()]))

    if config.run.do_train:
        print('training:', trainer)
        trainer.fit(model)

    if config.run.do_eval:
        print('evaluating...')

    if config.run.do_debug:
        print('debug actions...')
        sig = inspect.signature(MRPCTransformer)

    print('signature', sig)
    # config.model.model_name

if __name__ == '__main__':
    cli()