import argparse

import data_loader.data_loaders as module_data
from parse_config import ConfigParser
from trainer import COWCGANFrcnnTrainer

"""
python test.py -c config_GAN.json
"""


def main(config):
    data_loader = module_data.COWCGANFrcnnDataLoader(
        "/Users/vedaad/caliber/EESRGAN/data/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/valid_img/",
        "/Users/vedaad/caliber/EESRGAN/data/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/valid_img/",
        1,
        training=False,
    )
    tester = COWCGANFrcnnTrainer(config=config, data_loader=data_loader)
    tester.test()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args)
    main(config)
