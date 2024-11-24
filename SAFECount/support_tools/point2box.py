# coding=utf-8
# @Project  ：SAFECount 
# @FileName ：point2box.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/12/6 23:27
import json
import os
import os.path as osp
import random
import shutil
import sys

import cv2
from tqdm import tqdm

from support_tools.pascal_voc_utils import Writer
import numpy as np

def random_select_files(source_path, target_path, num_files = 100):
    os.makedirs(target_path, exist_ok=True)
    file_list = []
    for root, dirs, files in os.walk(source_path):
        for file in tqdm(files):
            if file.endswith('.jpg'):
                img_file = osp.join(root, file)
                json_file = osp.join(root, file.replace('.jpg', '.json'))
                if  osp.exists(json_file) and osp.exists(img_file):
                    file_list.append((osp.join(root, file), osp.join(root, file.replace('.jpg', '.json'))))
                else:
                    print(img_file, json_file, 'is not exist !')

    random.seed(0)
    random.shuffle(file_list)
    selected_files = file_list[:num_files]

    for (img_file, json_file) in tqdm(selected_files):
        shutil.move(img_file, target_path)
        shutil.move(json_file, target_path)

def generate_bbox(source_path, target_path, bbox_size = 10):
    os.makedirs(target_path, exist_ok=True)
    for root, dirs, files in os.walk(source_path):
        for file in tqdm(files):
            if file.endswith('.jpg'):
                img_file = osp.join(root, file)
                json_file = osp.join(root, file.replace('.jpg', '.json'))
                if  osp.exists(json_file) and osp.exists(img_file):
                    img = cv2.imread(img_file)
                    h, w = img.shape[:2]
                    with open(json_file, 'r') as f:
                        context = json.load(f)
                        filename, points = osp.splitext(context['filename'])[0], context['points']
                        writer = Writer(filename, w, h)
                        for (x, y) in points:
                            xmin, ymin, xmax, ymax = max(x - bbox_size, 0) , \
                                                     max(y - bbox_size, 0),  \
                                                     min(x + bbox_size, h), \
                                                     min(y + bbox_size, w)
                            writer.addObject('chicken', xmin, ymin, xmax, ymax)

                        writer.save(osp.join(target_path, filename + '.xml'))
                else:
                    print(img_file, json_file, 'is not exist !')




if __name__ == '__main__':
    # random_select_files('/Volumes/home/annotation/input', '/Volumes/home/annotation/input1')
    generate_bbox('/Volumes/home/annotation/input1', '/Volumes/home/annotation/input3_xml')