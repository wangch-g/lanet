import torch

from train import Trainer
from config import get_config
from utils import prepare_dirs
from data_loader import get_data_loader

def main(config):
    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.seed)
    if config.use_gpu:
        torch.cuda.manual_seed(config.seed)

    # instantiate train data loaders
    train_loader = get_data_loader(config=config)

    trainer = Trainer(config, train_loader=train_loader)
    trainer.train()

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)