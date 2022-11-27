import shutil
from pathlib import Path

import cv2
import pandas as pd
import PIL
from tqdm import trange
import torch
import numpy as np
import sys
sys.path.append("detectron2/detectron2")

import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

setup_logger()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

detectron = DefaultPredictor(cfg)

def detect(img):
    width, height, _ = img.shape

    outputs = detectron(img)

    classes = outputs["instances"].get_fields()["pred_classes"]
    bird_indices = np.where(classes == 14)[0]
    if len(bird_indices) == 0:
        left, top, right, bottom = 0, 0, width, height
    else:
        scores = outputs["instances"].get_fields()["scores"]
        for j in range(len(scores)):
            if j not in bird_indices:
                scores[j] = 0
        argmax = scores.argmax().item()
        box = outputs["instances"].get_fields()["pred_boxes"].tensor[argmax]
        left, top, right, bottom = box
        # left -= 10
        # right += 10
        # top -= 10
        # bottom += 10
        left, top, right, bottom = list(map(lambda x: int(x), (left, top, right, bottom)))
        
    return (left, top, right, bottom)

def crop(img, bbox):
    left, top, right, bottom = bbox
    img_crop = img[top:bottom, left:right]
    return img_crop

def extract_one(src_path, dst_path):
    img = cv2.imread(src_path)
    bbox = detect(img)
    img_crop = crop(img, bbox)
    cv2.imwrite(dst_path, img_crop)

def detect_all(src_folder):
    files = list(src_folder.rglob("*.jpg"))
    for i in trange(len(files)):
        f = files[i]
        try:
            extract_one(str(f), str(f))
        except:
            print(f"Error with img: {f} !")


if __name__ == "__main__":
    SOURCE = Path(sys.argv[1])
    
    detect_all(
        src_folder=SOURCE
    )