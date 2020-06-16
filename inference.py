import logging
import os
from collections import namedtuple

import torch

from parse_config import ConfigParser
from trainer import COWCGANFrcnnTrainer

logger = logging.getLogger(__name__)

config_name = "config_GAN.json"
config = ConfigParser(config_name, None, {})
pretrain_model_G_name = "170000_G.pth"
pretrain_model_D_name = "170000_D.pth"
pretrain_model_FRCNN_name = "170000_FRCNN.pth"


def model_fn(model_dir):
    # modify config to point to correct paths
    config["path"]["pretrain_model_G"] = os.path.join(model_dir, pretrain_model_G_name)
    config["path"]["pretrain_model_D"] = os.path.join(model_dir, pretrain_model_D_name)
    config["path"]["pretrain_model_FRCNN"] = os.path.join(
        model_dir, pretrain_model_FRCNN_name
    )
    # modify config n_gpu field
    config["n_gpu"] = os.environ.get("SM_NUM_GPUS", torch.cuda.device_count())

    # we do not need data loader since pre-trained models will be loaded from S3
    data_loader = None
    trainer = COWCGANFrcnnTrainer(config=config, data_loader=data_loader)
    return trainer.model


def input_fn(request_body, content_type="application/json"):
    return request_body


def predict_fn(input_data, model):
    # create data loader
    # pass dataloader into model.test or a closer endpoint
    pass


def output_fn(prediction, content_type="application/json"):
    return prediction
