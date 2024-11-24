# coding=utf-8
# @Project  ：SAFECount 
# @FileName ：synthetic_generator.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/2/10 18:51
import json
import os
import os.path as osp
import random
import shutil
import traceback
from collections import defaultdict
from multiprocessing.pool import ThreadPool

import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm

from data.Chicken.gen_gt_density import points2density, apply_scoremap
from support_tools.pascal_voc_utils import Reader


def load_chicken_patch(path, xml_path):
    patches = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('.jpg'):
                xml_name = osp.splitext(filename)[0] + '.xml'
                xml_file = os.path.join(xml_path, xml_name)
                if os.path.exists(xml_file):
                    img = cv2.imread(osp.join(root, filename))
                    objs = Reader(xml_file).get_objects()
                    for bbox in objs['bboxes']:
                        xmin, ymin, xmax, ymax = bbox
                        patch = img[ymin:ymax, xmin:xmax, :]
                        patches.append(patch)
    print('There has total %d patches' % len(patches))
    return patches


def load_bg_imgs(path):
    img_paths = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('jpg'):
                img_paths.append(osp.join(root, filename))
    print('Load %d background images' % len(img_paths))
    return img_paths

class SafeRescale(A.DualTransform):

    def __init__(self, min_size=16, scale_limit = 0.1, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super(SafeRescale, self).__init__(always_apply, p)
        self.scale_limit = A.to_tuple(scale_limit, bias=0.0)
        self.interpolation = interpolation
        self.min_size = min_size

    def get_params(self):
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale=0, interpolation=cv2.INTER_LINEAR, **params):
        height, width = img.shape[:2]
        new_height, new_width = max(int(height * scale), 1), max(int(width * scale), 1)
        min_scale = self.min_size / min(new_height, new_width)
        if min_scale > 1:
            new_height, new_width = int(new_height * min_scale), int(new_width * min_scale)
            # print('a', height, width, new_height, new_width)
            return A.resize(img, new_height, new_width, interpolation)
        else:
            # print('b', height, width, new_height, new_width)
            return A.resize(img, new_height, new_width, interpolation)





def get_aug_compose(type = 'patch'):
    assert type in ['patch', 'bg']
    if type == 'patch':
        aug = A.Compose([
            # A.OneOf([A.ElasticTransform(sigma=5, alpha_affine=5, p = 0.1),
            #                       A.OpticalDistortion(distort_limit=0.05, shift_limit=0, p =0.1)],p = 1),
                         A.Flip(p=0.5), A.Rotate(limit = (0, 270), p=0.5),
                         SafeRescale(min_size = 16, scale_limit=(0.6, 1.1), p = 1),
                         A.OneOf([A.Blur((3, 5), p =0.2), A.MedianBlur(blur_limit = 5, p = 0.2)], p = 1),
                         A.OneOf([A.RandomBrightnessContrast(p=0.2), A.ToGray(p=0.2)], p = 1),])
    else:
        aug = A.Compose([A.RandomCrop(width = 2200, height = 700, p = 1),
                         A.OpticalDistortion(distort_limit=0.05, shift_limit=0, p=0.2),
                         A.RandomBrightnessContrast(p = 0.5),
                         A.ToGray(p = 0.2),
                         A.Resize(700, 2200, p = 1)
                         ])
    return aug

def join_create_dir(*args):
    path = osp.join(*args)
    if '.' in osp.split(path)[-1]:
        folder_path = osp.split(path)[0]
    else:
        folder_path = path
    if not osp.exists(folder_path):
        os.makedirs(folder_path)
    return path

def img_synthetic(img_path, chicken_patches, index, output_path):
    try:
        patch_aug, bg_aug = get_aug_compose('patch'), get_aug_compose('bg')
        bg = cv2.imread(img_path)
        aug_bg = bg_aug(image = bg)['image']
        bg_height, bg_width = aug_bg.shape[:2]
        aug_patches = []
        max_height, max_width = -1, -1
        for patch in chicken_patches:
            patch = patch_aug(image = patch)['image']
            patch_size = patch.shape[:2]
            if max_height < patch_size[0]:
                max_height = patch_size[0]
            if max_width < patch_size[1]:
                max_width = patch_size[1]
            aug_patches.append(patch)

        random_h = random.choices(range(0, bg_height - max_height - 1), k = len(aug_patches))
        random_w = random.choices(range(0, bg_width - max_width - 1), k = len(aug_patches))
        json_content = {'filename': '%08d.jpg' % index, 'density': '%08d.npy' % index, 'points':[]}
        for h, w, patch in zip(random_h, random_w, aug_patches):
            aug_bg[h:h + patch.shape[0], w:w + patch.shape[1], :] = patch
            center_point = [(w + w+patch.shape[1]) // 2, (h + h+patch.shape[0]) // 2]
            json_content['points'].append(center_point)

        # img
        cv2.imwrite(join_create_dir(output_path, 'frames', '%08d.jpg' % index), aug_bg)
        # json with point annotation
        with open(join_create_dir(output_path, 'frames', '%08d.json' % index), 'w', encoding='utf-8') as f:
            json.dump(json_content, f, indent=4)


        # draw heatmap
        points = np.array(json_content['points'])
        cnt_gt = points.shape[0]

        density = points2density(points, max_scale=3.0, max_radius=15.0, image_size=(bg_height, bg_width))
        if not cnt_gt == 0:
            cnt_cur = density.sum()
            density = density / cnt_cur * cnt_gt
        # print(np.sum(density), len(points))
        np.save(join_create_dir(output_path, 'gt_density_map', '%08d.npy' % index), density)


        ## draw vis
        # min, max = density.min(), density.max()
        # density_norm = (density - min) / (max - min + 1e-8)
        # aug_bg_copy = cv2.cvtColor(aug_bg, cv2.COLOR_BGR2RGB)
        # density_vis = apply_scoremap(aug_bg_copy, density_norm, 0.5)
        # density_vis = cv2.cvtColor(density_vis, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(join_create_dir(output_path, 'vis', '%08d.jpg' % index), density_vis)
    except Exception as e:
        print('img_synthetic error', e, traceback.format_exc())

        npy_path = join_create_dir(output_path, 'gt_density_map', '%08d.npy' % index)
        img_path = join_create_dir(output_path, 'frames', '%08d.jpg' % index)
        json_path = join_create_dir(output_path, 'frames', '%08d.json' % index)

        if osp.exists(npy_path):
            os.remove(npy_path)

        if osp.exists(img_path):
            os.remove(img_path)

        if osp.exists(json_path):
            os.remove(json_path)


def generate_synthetic_data(chicken_patch_path, xml_path, bg_path, output_path, num_imgs=1000, num_chicken_per_img = 200,
                            seed =
1, start_index = 0):
    os.makedirs(output_path, exist_ok=True)
    chicken_patches = load_chicken_patch(chicken_patch_path, xml_path)
    bg_img_paths = load_bg_imgs(bg_path)
    random.seed(seed)
    pool = ThreadPool(100)
    bar = tqdm(total=num_imgs, desc='total')
    update = lambda *args: bar.update()

    for i in range(start_index, num_imgs + start_index):
        try:
            img_path = bg_img_paths[random.randint(0, len(bg_img_paths) - 1)]
            chicken_index = random.choices(range(0, len(chicken_patches)), k = random.randint(1, num_chicken_per_img))
            patches = [chicken_patches[index] for index in chicken_index]
            # img_synthetic(img_path, patches, i, output_path)

            kwds = {'img_path': img_path, 'chicken_patches': patches, 'index': i, 'output_path': output_path}
            pool.apply_async(img_synthetic, kwds=kwds, callback=update)
        except Exception as e:
            print(e)
            continue

    pool.close()
    pool.join()


def check_validate(output_path):
    files = defaultdict(list)
    for root, _, filenames in os.walk(output_path):
        for filename in filenames:
            if 'vis' not in root:
                name = osp.splitext(filename)[0]
                files[name].append(osp.join(root, filename))

    for name, paths in tqdm(files.items(), total = len(files.keys())):
        try:
            if len(paths) != 3:
                assert ValueError('There must has three files')
                # for p in paths:
                #     # os.remove(p)
                #     assert ValueError('There must has three files')
            else:
                for p in paths:
                    if 'jpg' in p:
                        img = cv2.imread(p)
                        h, w, _ = img.shape
                        assert h == 700 and w == 2200, 'img size error %s' % p.shape
                    elif 'npy' in p:
                        heatmap = np.load(p)
                        h, w = heatmap.shape
                        assert h == 700 and w == 2200, 'np.shape: %s' % heatmap.shape
                    elif 'json' in p:
                        with open(p, 'r', encoding='utf-8') as f:
                            content = json.load(f)
                            assert len(content['points']) > 0, 'points is empty'
                    else:
                        assert TypeError('File type error, got %s' % p)
        except Exception as e:
            print(e, traceback.format_exc())
            for p in paths:
                # print('file')
                os.remove(p)
                print(p)

def copy_files(src_path, dst_path, json_path):
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            content = json.loads(line)
            name = osp.splitext(content['filename'])[0]
            img = osp.join(src_path, 'frames', name + '.jpg')
            json_obj = osp.join(src_path, 'frames', name + '.json')
            npy = osp.join(src_path, 'gt_density_map', name + '.npy')

            if osp.exists(img) and osp.exists(json_obj) and osp.exists(npy):
                shutil.copy(img, osp.join(dst_path, 'frames'))
                shutil.copy(json_obj, osp.join(dst_path, 'frames'))
                shutil.copy(npy, osp.join(dst_path, 'gt_density_map'))
            else:
                print('File not exists', img, json, npy)


if __name__ == '__main__':
    # generate_synthetic_data(
    #     chicken_patch_path='/Volumes/SoberSSD/SSD_Download/chicken/chicken_2_finetune/camera_test/frames',
    #     xml_path='/Volumes/SoberSSD/SSD_Download/chicken/chicken_2_finetune/camera_test/xml',
    #     bg_path='/Volumes/SoberSSD/SSD_Download/chicken/no_chicken_clip',
    #     output_path='/Volumes/SoberSSD/SSD_Download/chicken/no_chicken_clip_synthetic',
    #     num_imgs=10000,
    #     num_chicken_per_img=100,
    #     start_index=40001,
    # )
    # generate_synthetic_data(
    #     chicken_patch_path='/Volumes/SoberSSD/SSD_Download/chicken/chicken_count_process/input',
    #     xml_path='/Volumes/SoberSSD/SSD_Download/chicken/chicken_count_process/xml',
    #     bg_path='/Volumes/SoberSSD/SSD_Download/chicken/no_chicken_clip',
    #     output_path='/Volumes/SoberSSD/SSD_Download/chicken/no_chicken_clip_synthetic_tp', num_imgs=20,
    #     num_chicken_per_img=200, )

    # check_validate('/Volumes/SoberSSD/SSD_Download/chicken/no_chicken_clip_synthetic')

    copy_files('/Volumes/SoberSSD/SSD_Download/chicken/chicken_origin/',
               '/Volumes/SoberSSD/SSD_Download/chicken/no_chicken_clip_synthetic', '/Users/sober/Workspace/Python/SAFECount/data/Chicken/camera/train.json')