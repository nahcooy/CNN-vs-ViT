# coding=utf-8
# @Project  ：SAFECount 
# @FileName ：data_split.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/1/27 18:49
import argparse
import copy
import json
import os
import os.path as osp
import random
import sys

import cv2


sys.path.append('./')
sys.path.append('../')
from data.Chicken.gen_gt_density import points2density, apply_scoremap
import numpy as np
from tqdm import tqdm
from support_tools.pascal_voc_utils import Reader

def parse():
    parser = argparse.ArgumentParser(description='SAFECount data split')
    parser.add_argument('--input_dir', required=True, help='input directory with json files')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--xml_dir', required=True, help='where the test xml file saved')
    #
    # parser.add_argument('--test_suffix', type=str, default='xml', help='If the file has xml annotation, it is a test set')
    # parser.add_argument('--split_type', type=str, default='test', help='random split')

    # parser.add_argument('--input_dir', default='/Volumes/SoberSSD/SSD_Download/chicken/demo/query/frames',
    #                     help='input directory with json files')
    # parser.add_argument('--output_dir', default='/Volumes/SoberSSD/SSD_Download/chicken/demo',
    #                     help='output directory')
    # parser.add_argument('--xml_dir', default='/Volumes/SoberSSD/SSD_Download/chicken/demo/query/xml', help='where '
    #                                                                                                         'the test xml file saved')
    parser.add_argument('--test_suffix', type=str, default='xml',
                        help='If the file has xml annotation, it is a test set')
    parser.add_argument('--split_type', type=str, default='random', help='random split')

    args = parser.parse_args()
    print(args)
    return args

# def json_generator(files, root_dir, target_path):
#     total_f = open(target_path, 'w+', encoding='utf-8')
#     for file in tqdm(files, desc=target_path):
#         objs = None
#         if len(file) == 2:
#             file, objs = file
#         filename = osp.join(root_dir, file)
#         with open(filename, 'r+', encoding='utf-8') as f:
#             content = json.load(f)
#             if objs is not None:
#                 content['boxes'] = []
#                 for bbox in objs:
#                     xmin, ymin, xmax, ymax = bbox
#                     content['boxes'].append([ymin, xmin, ymax, xmax])
#             content['filename'] = osp.splitext(content['filename'])[0] + '.jpg'
#             content = json.dumps(content)
#             total_f.write(content + '\n')
#     total_f.close()


def draw_density_vis(path, points, target_path, vis_heatmap = True):
    if osp.exists(path):
        name = osp.splitext(osp.split(path)[-1])[0]
        img = cv2.imread(path)
        cnt_gt = points.shape[0]
        density = points2density(points, max_scale=3.0, max_radius=15.0, image_size=img.shape[:2])
        if not cnt_gt == 0:
            cnt_cur = density.sum()
            density = density / cnt_cur * cnt_gt

        os.makedirs(osp.join(target_path, 'gt_density_map'), exist_ok=True)
        np.save(osp.join(target_path, 'gt_density_map', name + '.npy'), density)

        if vis_heatmap:
            os.makedirs(osp.join(target_path, 'vis'), exist_ok=True)
            min, max = density.min(), density.max()
            density = (density - min) / (max - min + 1e-8)
            mask = apply_scoremap(img, density)
            cv2.imwrite(os.path.join(target_path, 'vis', name + '.jpg'), mask)

        return True
    else:
        return False

def json_generator(files, root_dir, target_path):

    total_f = open(target_path, 'w+', encoding='utf-8')
    for file in tqdm(files, desc=target_path):
        objs = None
        if len(file) == 2:
            file, objs = file
        filename = osp.join(root_dir, file)
        name = osp.splitext(file)[0]
        if filename.endswith('json'):
            with open(filename, 'r+', encoding='utf-8') as f:
                content = json.load(f)
                if objs is not None:
                    content['boxes'] = []
                    for bbox in objs:
                        xmin, ymin, xmax, ymax = bbox
                        content['boxes'].append([ymin, xmin, ymax, xmax])
                content['filename'] = osp.splitext(content['filename'])[0] + '.jpg'
                content = json.dumps(content)
                total_f.write(content + '\n')
        # xml end with jpg
        else:
            if objs is not None:
                content = {'filename':name + '.jpg', 'density':name + '.npy', 'points':[]}
                for bbox in objs:
                    xmin, ymin, xmax, ymax = bbox
                    # content['boxes'].append([ymin, xmin, ymax, xmax])
                    content['points'].append([(xmax+xmin)/2.0, (ymax+ymin)/2.0])

                points = np.array(content['points'])
                if draw_density_vis(filename, points, root_dir, vis_heatmap=True):
                    content = json.dumps(content)
                    total_f.write(content + '\n')
    total_f.close()



if __name__ == '__main__':
    args = parse()
    os.makedirs(args.output_dir, exist_ok=True)

    items,  xml_items = [], []
    for root, _, filenames in os.walk(args.input_dir):
        for filename in sorted(tqdm(filenames, desc='Read xmls')):
            if filename.endswith('jpg')  and not filename.startswith('.'):
                name = osp.splitext(filename)[0]
                # has xml file, this image can be test image

                if osp.exists(osp.join(root, filename.replace('jpg', 'json'))):
                    filename = filename.replace('jpg', 'json')
                if osp.exists(osp.join(args.xml_dir, name+'.xml')):
                    xml_p = osp.join(args.xml_dir, name+'.xml')
                    objs = Reader(xml_p).get_objects()['bboxes']
                    xml_items.append((filename, objs))
                else:
                    items.append(filename)

    assert args.split_type in ['random', 'xml', 'exemplar', 'test', 'train', 'exemplar_xml']

    if args.split_type == 'random':
        # random select some item for train and test, with rate 4:1
        items.extend(xml_items)

        random.seed(0)
        random.shuffle(items)

        test_items = items[:int(len(items)*0.2)]
        train_items = items[int(len(items)*0.2):]

    elif args.split_type == 'exemplar':
        #the xml item must be test data
        random.seed(0)
        random.shuffle(items)

        test_items = items[:int(len(items) * 0.2)]
        test_items.extend(xml_items)
        train_items = items[int(len(items) * 0.2):]

        f = open(osp.join(args.output_dir, 'exemplar.json'), 'w+', encoding='utf-8')

        for xml_item in xml_items:
            filename, objs = xml_item
            xmin, ymin, xmax, ymax = objs[0]
            content = json.dumps({'filename': osp.splitext(filename)[0] + '.jpg', 'box': [ymin, xmin, ymax, xmax]})
            f.write(content + '\n')
        f.close()

    elif args.split_type == 'xml':
        # only the image has xml file can be for train/test
        random.seed(0)
        random.shuffle(items)

        test_items = items[:int(len(items) * 0.2)]
        train_items = items[int(len(items) * 0.2):]
    elif args.split_type == 'test':
        f = open(osp.join(args.output_dir, 'exemplar_test.json'), 'w+', encoding='utf-8')

        for xml_item in xml_items:
            filename, objs = xml_item
            xmin, ymin, xmax, ymax = objs[0]
            content = json.dumps({'filename': osp.splitext(filename)[0] + '.jpg', 'box': [ymin, xmin, ymax, xmax]})
            f.write(content + '\n')
        f.close()
        json_generator(xml_items, args.input_dir, osp.join(args.output_dir, 'test_test.json'))
        sys.exit(0)
    elif args.split_type == 'train':
        json_generator(items, args.input_dir, osp.join(args.output_dir, 'train.json'))
        sys.exit(0)
    elif args.split_type == 'exemplar_xml':
        # labelme xml annotation
        # because all the point annotation is xml based, so
        items.extend(xml_items)
        exemplar_items = items[:50]
        items = items[50:]
        random.seed(0)
        random.shuffle(items)

        test_items = items[:int(len(items) * 0.2)]
        train_items = items[int(len(items) * 0.2):]

        f = open(osp.join(args.output_dir, 'exemplar.json'), 'w+', encoding='utf-8')

        for xml_item in exemplar_items:
            filename, objs = xml_item
            xmin, ymin, xmax, ymax = objs[0]
            content = json.dumps({'filename': osp.splitext(filename)[0] + '.jpg', 'box': [ymin, xmin, ymax, xmax],
                                  'points':[abs(ymin - ymax) // 2, abs(xmin - xmax) // 2]})
            f.write(content + '\n')
        f.close()
    else:
        raise ValueError('Split type error')


    json_generator(train_items, args.input_dir, osp.join(args.output_dir, 'train.json'))
    json_generator(test_items, args.input_dir, osp.join(args.output_dir, 'test.json'))