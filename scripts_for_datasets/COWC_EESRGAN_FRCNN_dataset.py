from __future__ import print_function, division

import glob
import logging
import os
import warnings

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")
logger = logging.getLogger("base")

class COWCGANFrcnnDataset(Dataset):
    def __init__(
        self,
        data_dir_gt,
        data_dir_lq,
        image_height=256,
        image_width=256,
        transform=None,
        inference=False,
        inference_request=None
    ):
        self.data_dir_gt = data_dir_gt
        self.data_dir_lq = data_dir_lq
        # take all under same folder for train and test split.
        self.transform = transform
        self.image_height = image_height
        self.image_width = image_width
        # sort all images for indexing, filter out check.jpgs

        # if doing inference, we just take one image
        self.inference = inference
        if not inference:
            logger.info(f"in cowcganfrcnndataset now")
            logger.info(f"data_dir_gt: {data_dir_gt}")
            logger.info(f"data_dir_lq: {data_dir_lq}")
            self.imgs_gt = list(sorted(glob.glob(self.data_dir_gt + "*.jpg")))
            self.imgs_lq = list(sorted(glob.glob(self.data_dir_lq + "*.jpg")))
            self.annotation = list(sorted(glob.glob(self.data_dir_lq + "*.txt")))

            logger.info(f"len(imgs_gt): {len(self.imgs_gt)}")
            logger.info(f"len(imgs_lq): {len(self.imgs_lq)}")
            logger.info(f"len(annotation): {len(self.annotation)}")
        else:
            self.imgs_lq = [""] # this is a hack to make the __len__ function work
            self.inference_request = inference_request

    def getitem_inference(self, idx):
        # expects a cv2 object
        img_gt = img_lq = cv2.cvtColor(self.inference_request, cv2.COLOR_BGR2RGB)

        # create dictionary to access the values
        target = {}
        target["object"] = 0
        target["image_lq"] = img_lq
        target["image"] = img_gt
        target["bboxes"] = [[0, 0, 1, 1]]
        target["labels"] = [1]
        target["label_car_type"] = [0]
        target["image_id"] = idx
        target["LQ_path"] = None
        target["area"] = [0]
        target["iscrowd"] = [0]

        if self.transform is not None:
            target = self.transform(**target)

        image, target = self.convert_to_tensor(**target)
        return image, target

    def __getitem__(self, idx):
        if self.inference:
            # this is a function that returns the image in the format that is required but not the target
            # this function also does not return any info about high quality image
            return self.getitem_inference(idx)

        # get the paths
        img_path_gt = os.path.join(self.data_dir_gt, self.imgs_gt[idx])
        img_path_lq = os.path.join(self.data_dir_lq, self.imgs_lq[idx])
        annotation_path = os.path.join(self.data_dir_lq, self.annotation[idx])
        img_gt = cv2.imread(img_path_gt, 1)  # read color image height*width*channel=3
        img_lq = cv2.imread(img_path_lq, 1)  # read color image height*width*channel=3
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

        # get the bounding box
        boxes = list()
        label_car_type = list()
        with open(annotation_path) as f:
            for line in f:
                values = line.split()
                if "\ufeff" in values[0]:
                    values[0] = values[0][-1]
                obj_class = int(values[0])
                # image without bounding box - in txt file, line starts with 0 and only contains only 0
                if obj_class == 0:
                    boxes.append([0, 0, 1, 1])
                    labels = np.ones(len(boxes))  # all are cars
                    label_car_type.append(obj_class)
                    # create dictionary to access the values
                    target = {}
                    target["object"] = 0
                    target["image_lq"] = img_lq
                    target["image"] = img_gt
                    target["bboxes"] = boxes
                    target["labels"] = labels
                    target["label_car_type"] = label_car_type
                    target["image_id"] = idx
                    target["LQ_path"] = img_path_lq
                    target["area"] = 0
                    target["iscrowd"] = 0
                    break
                else:
                    # get coordinates withing height width range
                    x = float(values[1]) * self.image_width
                    y = float(values[2]) * self.image_height
                    width = float(values[3]) * self.image_width
                    height = float(values[4]) * self.image_height
                    # creating bounding boxes that would not touch the image edges
                    x_min = 1 if x - width / 2 <= 0 else int(x - width / 2)
                    x_max = 255 if x + width / 2 >= 256 else int(x + width / 2)
                    y_min = 1 if y - height / 2 <= 0 else int(y - height / 2)
                    y_max = 255 if y + height / 2 >= 256 else int(y + height / 2)

                    boxes.append([x_min, y_min, x_max, y_max])
                    label_car_type.append(obj_class)

        if obj_class != 0:
            labels = np.ones(len(boxes))  # all are cars
            boxes_for_calc = torch.as_tensor(boxes, dtype=torch.int64)
            area = (boxes_for_calc[:, 3] - boxes_for_calc[:, 1]) * (
                boxes_for_calc[:, 2] - boxes_for_calc[:, 0]
            )
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
            # create dictionary to access the values
            target = {}
            target["object"] = 1
            target["image_lq"] = img_lq
            target["image"] = img_gt
            target["bboxes"] = boxes
            target["labels"] = labels
            target["label_car_type"] = label_car_type
            target["image_id"] = idx
            target["LQ_path"] = img_path_lq
            target["area"] = area
            target["iscrowd"] = iscrowd


        if self.transform is None:
            # convert to tensor
            image, target = self.convert_to_tensor(**target)
            return image, target
            # transform
        else:
            transformed = self.transform(**target)
            image, target = self.convert_to_tensor(**transformed)
            return image, target

    def __len__(self):
        return len(self.imgs_lq)

    def convert_to_tensor(self, **target):
        # convert to tensor
        target["object"] = torch.tensor(target["object"], dtype=torch.int64)
        target["image_lq"] = torch.from_numpy(target["image_lq"].transpose((2, 0, 1)))
        target["image"] = torch.from_numpy(target["image"].transpose((2, 0, 1)))
        target["boxes"] = torch.tensor(target["bboxes"], dtype=torch.float32)
        target["labels"] = torch.ones(len(target["labels"]), dtype=torch.int64)
        target["label_car_type"] = torch.tensor(
            target["label_car_type"], dtype=torch.int64
        )
        target["image_id"] = torch.tensor([target["image_id"]])
        target["area"] = torch.tensor(target["area"])
        target["iscrowd"] = torch.tensor(target["iscrowd"])

        image = {}
        image["object"] = target["object"]
        image["image_lq"] = target["image_lq"]
        image["image"] = target["image"]
        image["image"] = target["image"]
        image["LQ_path"] = target["LQ_path"]

        del target["object"]
        del target["image_lq"]
        del target["image"]
        del target["bboxes"]
        del target["LQ_path"]

        return image, target
