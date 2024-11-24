# coding=utf-8
# @Project  ：SAFECount 
# @FileName ：safe_data_convert.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/1/27 18:22
import argparse
import sys
sys.path.append('./')
sys.path.append('../')
import json
import os
import os.path as osp
from tqdm import tqdm
from support_tools.pascal_voc_utils import Reader


def parse():
    parser = argparse.ArgumentParser(description='SAFECount')
    parser.add_argument('--input_dir', required=True, help='input directory')
    parser.add_argument('--output_dir', required=True, help='output directory')
    args = parser.parse_args()
    print(args)
    return args

def read_voc_xml(path):
    items = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.xml'):
                name = osp.splitext(filename)[0]
                reader = Reader(osp.join(root, filename))
                objs = reader.get_objects()
                bboxes =  objs['bboxes']
                assert  len(bboxes) > 0
                for bbox in bboxes:
                    items.append({'filename': name+'.jpg', 'bboxes': bbox})

    return items

def to_json(items, target_path):
    os.makedirs(target_path, exist_ok=True)
    f = open(osp.join(target_path, 'exemplar.json'), 'w', encoding='utf-8')

    for item in tqdm(items, desc='To json'):
        content = json.dumps(item)
        f.write(content + '\n')

    f.close()



if __name__ == '__main__':
    args = parse()
    items = read_voc_xml(args.input_dir)
    to_json(items, args.output_dir)
