import glob
import logging
import math
import os

import numpy as np
import torch

import data_loader.data_loaders as module_data
from parse_config import ConfigParser
from trainer import COWCGANFrcnnTrainer
from utils import setup_logger
from utils.util import read_json

"""
python train.py -c config_GAN.json
"""

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, logger):
    # logger = config.get_logger('train')
    # setup data_loader instances
    data_loader = config.init_obj("data_loader", module_data)
    # change later this valid_data_loader using init_obj

    val_gt_dir = os.path.join(os.environ["SM_CHANNEL_VAL"], "HR")
    val_lq_dir = os.path.join(os.environ["SM_CHANNEL_VAL"], "LR")
    valid_data_loader = module_data.COWCGANFrcnnDataLoader(
        val_gt_dir,
        val_lq_dir,
        # "/Users/vedaad/caliber/EESRGAN/data/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/valid_img/",
        # "/Users/vedaad/caliber/EESRGAN/data/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/valid_img/",
        1,
        training=False,
    )

    logger.info(f"Val channels: {val_gt_dir}, {val_lq_dir}")

    trainer = COWCGANFrcnnTrainer(
        config=config, data_loader=data_loader, valid_data_loader=valid_data_loader
    )
    trainer.train()


if __name__ == "__main__":
    N_EPOCHS = 3

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(curr_dir, "config_GAN.json")
    config_obj = read_json(config_path)
    config = ConfigParser(config_obj, None, {})

    # set train data location config vars
    train_gt_dir = os.path.join(os.environ["SM_CHANNEL_TRAIN"], "HR")
    train_lq_dir = os.path.join(os.environ["SM_CHANNEL_TRAIN"], "LR")
    config["data_loader"]["args"]["data_dir_GT"] = train_gt_dir
    config["data_loader"]["args"]["data_dir_LQ"] = train_lq_dir

    # set number of iterations to complete 3 epochs
    n_training_images = len(glob.glob(os.path.join(train_gt_dir, "*.jpg")))
    batch_size = config["data_loader"]["args"]["batch_size"]
    n_iters = math.ceil(n_training_images / batch_size * N_EPOCHS)
    config["train"]["niter"] = n_iters

    # set model save location
    config["path"]["models"] = os.environ["SM_MODEL_DIR"]

    # set num gpus
    config["n_gpu"] = os.environ.get("SM_NUM_GPUS", torch.cuda.device_count())

    # set log output location
    config["path"]["log"] = os.environ["SM_OUTPUT_DATA_DIR"]

    # config logger
    setup_logger(
        "base",
        config["path"]["log"],
        "train_" + config["name"],
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    setup_logger(
        "val",
        config["path"]["log"],
        "val_" + config["name"],
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")

    # log number of training images
    n_training_images_lq = len(glob.glob(os.path.join(train_lq_dir, "*.jpg")))
    logger.info(f"Train channels: {train_gt_dir}, {train_lq_dir}")
    logger.info(f"n_training_images_gt: {n_training_images}")
    logger.info(f"n_training_images_lq: {n_training_images_lq}")

    main(config, logger)
