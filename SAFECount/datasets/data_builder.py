import torch.distributed as dist

from datasets.custom_dataset import build_custom_dataloader
from datasets.custom_exemplar_dataset import build_custom_exemplar_dataloader
import numpy as np

def build(cfg, dataset_type, distributed):
    if dataset_type == "train":
        cfg.update(cfg.get("train", {}))
        training = True
    elif dataset_type == "val":
        cfg.update(cfg.get("val", {}))
        training = False
    elif dataset_type == "test":
        cfg.update(cfg.get("test", {}))
        training = False
    else:
        raise ValueError("dataset_type must among [train, val, test]!")

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed)
    elif dataset == "custom_exemplar":
        data_loader = build_custom_exemplar_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=True):
    if distributed:
        rank = dist.get_rank()
    else:
        rank = 0

    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, dataset_type="train", distributed=distributed)

    val_loader = None
    if cfg_dataset.get("val", None):
        val_loader = build(cfg_dataset, dataset_type="val", distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, dataset_type="test", distributed=distributed)

    if rank == 0:
        print("build dataset done")

    return train_loader, val_loader, test_loader


def matlab_style_gauss2D(shape=(3,3),sigma=8):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def covert_FSC147_trainValTest(json_path, target_path, split_rate= [0.8, 0.1, 0.1]):
    import json, random, os

    import os.path as osp

    assert sum(split_rate) == 1
    meta = []
    with open(json_path, 'r') as f:
        json_content = json.load(f)
        for k, v in json_content.items():
            boxes = []
            for bbox in v['box_examples_coordinates']:
                bbox = np.asarray(bbox)
                # print(bbox)
                xmin, ymin, xmax, ymax =  bbox.min(0)[0], bbox.min(0)[1], bbox.max(0)[0], bbox.max(0)[1]
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                boxes.append([xmin, ymin, xmax, ymax])

            item = {'filename':k, 'density':k.replace('jpg', 'npy'), 'boxes':boxes}
            meta.append(item)

    random.shuffle(meta)
    total = len(meta)
    train_meta = meta[: int(total*split_rate[0])]
    val_meta = meta[int(total*split_rate[0]):int(total*split_rate[1]) + int(total*split_rate[0])]
    test_meta = meta[int(total*split_rate[1]) + int(total*split_rate[0]):]

    os.makedirs(target_path, exist_ok=True)
    with open(osp.join(target_path, 'train.json'), 'w+') as f:
        for meta in train_meta:
            f.write(json.dumps(meta, indent = 4) + '\n')

    with open(osp.join(target_path, 'val.json'), 'w+') as f:
        for meta in val_meta:
            f.write(json.dumps(meta, indent = 4) + '\n')

    with open(osp.join(target_path, 'test.json'), 'w+') as f:
        for meta in test_meta:
            f.write(json.dumps(meta, indent = 4) + '\n')


if __name__ == '__main__':
    p1 = '/Volumes/datasets/LearningToCountEverything/data/annotation_FSC147_384.json'
    p2 = '/Volumes/datasets/counting'
    covert_FSC147_trainValTest(p1, p2)