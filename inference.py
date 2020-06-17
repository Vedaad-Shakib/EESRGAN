import logging
import os
import urllib

import cv2
import numpy as np
import torch

import data_loader.data_loaders as module_data
from parse_config import ConfigParser
from trainer import COWCGANFrcnnTrainer
from utils.util import read_json

logger = logging.getLogger(__name__)

config_name = "config_GAN.json"
config_obj = read_json(config_name)
config = ConfigParser(config_obj, None, {})
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
    """request body is a URL"""

    if content_type != "application/json":
        raise Exception(f'Requested unsupported ContentType: {content_type}')

    req = urllib.request.urlopen(request_body)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, 1)

    data_loader = module_data.COWCGANFrcnnDataLoader(
        None,
        None,
        1,
        training=False,
        inference=True,
        inference_request=img,
    )
    return data_loader


def predict_fn(data_loader, model):
    # modify it so it just returns the bounding boxes
    # modify it so it does not perform validation with the .txt shit
    result = model.test(data_loader, train=False, testResult=True, inference=True)
    return result


def output_fn(prediction, content_type="application/json"):
    return prediction
