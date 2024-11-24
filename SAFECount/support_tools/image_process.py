# coding=utf-8
# @Project  ：SAFECount 
# @FileName ：image_process.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/2/4 4:26 下午
import json
import multiprocessing
from multiprocessing.pool import ThreadPool
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os.path as osp
import os

from data.Chicken.gen_gt_density import apply_scoremap, points2density


def show_image(img, title='', cmap = 'gray'):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()

def reshape_points(points, a, b, c, d):
    tp_points = []
    for (x, y) in points:
        if x > d or y > b:
            continue
        else:
            x = x - c
            y = y - a
            if x < 0 or y < 0:
                continue
            else:
                tp_points.append((x, y))
    return tp_points

def save2file(target_path, filename, img, points, density, vis_heatmap = False):
    gt_density_map = osp.join(target_path, 'gt_density_map')
    frames = osp.join(target_path, 'frames')
    os.makedirs(gt_density_map, exist_ok=True)
    os.makedirs(frames, exist_ok=True)

    name = osp.splitext(filename)[0]
    json_content = {'filename':name + '.jpg', 'density': name + '.npy',
     'points':
         points.tolist()}
    with open(osp.join(frames, name+'.json'), 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=4)

    np.save(osp.join(gt_density_map, name+'.npy'), density)

    cv2.imwrite(os.path.join(frames, name + '.jpg'), img)

    if vis_heatmap:
        min, max = density.min(), density.max()
        density = (density - min) / (max - min + 1e-8)
        mask = apply_scoremap(img, density)
        cv2.imwrite(os.path.join(gt_density_map, name + '_vis.jpg'), mask)


def find_line(path):
    img = cv2.imread(path)
    height, width, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    kernel = np.ones((9, 9), np.uint8)
    erode = cv2.erode(img_blur, kernel, iterations=1)


    ret, thresh = cv2.threshold(erode, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = 255 - thresh

    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=(3, 3), iterations=0)
    show_image(close, 'close')

    # minLineLength = 2500
    # maxLineGap = 800
    minLineLength = 2500
    maxLineGap = 500
    lines = cv2.HoughLinesP(close, 1, np.pi / 180, 1000, minLineLength, maxLineGap)
    print(lines.shape)

    rm_line = img.copy()
    pipe_raw = np.zeros_like(img, dtype=np.uint8)
    pipe_blur = np.zeros((height, width), np.uint8)

    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        # if abs(y1 - y2) ^ 2 + abs(x1 - x2) ^ 2 > 0:
        #     rm_line = cv2.line(rm_line, (x1, y1), (x2, y2), (255, 0, 255), 5)
        rm_line = cv2.line(rm_line, (0, y1), (width, y2), (255, 0, 255), 5)
        #
        pipe_raw[y1-200:y2+200, 0:width] = img[y1-200:y2+200, 0:width]
        pipe_blur[y1 - 200:y2 + 200, 0:width] = img_blur[y1 - 200:y2 + 200, 0:width]




    # show_image(sobelx, 'sobelx')
    # show_image(sobelxy, 'sobelxy')
    # show_image(edges)
    # show_image(thresh)
    # show_image(close, 'close')
    show_image(rm_line, 'line', cmap=None)
    show_image(pipe_raw, 'crop', cmap=None)

    return pipe_raw, pipe_blur

# def find_line(path):
#     img = cv2.imread(path)
#     height, width, _ = img.shape
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img_blur = cv2.GaussianBlur(gray, (3, 3), 0)
#
#     fx, fy = 4137.8, 4147.3
#     K = np.array([[fx, 0, height//2], [0, fy, width//2], [0, 0, 1]])
#     dist_coeff = np.array([k1, k2, p1, p2, k3])
#
#     img_shape = gray.shape[::-1]
#     new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeff, img_shape, 1, img_shape)
#
#     dst = cv2.undistort(img, K, dist_coeff, None, new_K)
#     show_image(dst, 'dst', cmap=None)

def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if len(cnt) > 100:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img, ellipse, (0, 255, 0), 2)
    show_image(thresh)
    show_image(img, 'img', cmap=None)

def operation(root, filename, target_path):
    img = cv2.imread(osp.join(root, filename))


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (5,5))

    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh = 255 - thresh
    kernel = np.ones((9,9),np.uint8)
    erode = cv2.erode(thresh, kernel, iterations = 1)



    minLineLength = 500
    maxLineGap = 10
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 100, minLineLength, maxLineGap)

    rm_line = erode.copy()
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = line
        if abs(y1 - y2)^2 + abs(x1 - x2)^2 > 70:
            rm_line = cv2.line(rm_line, (x1, y1), (x2, y2), (0, 0, 0), 5)

    rm_line =cv2.blur(rm_line, (9,9))
    contours, hierarchy = cv2.findContours(rm_line, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_contours, points = [], []
    for i, contour in enumerate(contours):
        # ignore the small polygon
        if len(contour) < 10 or cv2.contourArea(contour) < 8*8:
            continue
        contour = np.array(contour)
        final_contours.append(contour)
        points.append([contour[:, :, 0].mean(), contour[:, :, 1].mean()])


    a, b, c, d = 300, 1000, 300, 2500
    img = img[a:b, c:d, :]

    points = reshape_points(points, a, b, c, d)

    points = np.array(points)

    cnt_gt = points.shape[0]
    density = points2density(points, max_scale=3.0, max_radius=15.0, image_size = img.shape[:2])

    if not cnt_gt == 0:
        cnt_cur = density.sum()
        density = density / cnt_cur * cnt_gt


    save2file(target_path, filename, img, points, density, True)


def test(height = 1000, width = 2000):
    import numpy as np
    import matplotlib.pyplot as plt

    def generate_rectangles(n, a, b):
        rectangles = []
        for i in range(n):
            w, h = np.random.randint(a, b + 1, size=2)
            x, y = np.random.randint(0, width - w + 1), np.random.randint(0, height - h + 1)
            rectangles.append((x, y, w, h))
        return rectangles

    def check_overlap(rectangles):
        for i, rect1 in enumerate(rectangles):
            for j, rect2 in enumerate(rectangles[i + 1:], i + 1):
                if (rect1[0] < rect2[0] + rect2[2] and rect1[0] + rect1[2] > rect2[0] and rect1[1] < rect2[1] + rect2[
                    3] and rect1[1] + rect1[3] > rect2[1]):
                    return True
        return False

    def draw_rectangles(rectangles):
        fig, ax = plt.subplots(figsize=(10, 10))
        for rect in rectangles:
            ax.add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], facecolor='blue', alpha=0.5))
        plt.xlim([0, 1000])
        plt.ylim([0, 1000])
        plt.axis('off')
        plt.show()

    n = 500  # 长方形数量
    a = 16  # 最小边长
    b = 48  # 最大边长

    while True:
        rectangles = generate_rectangles(n, a, b)
        if not check_overlap(rectangles):
            break

    draw_rectangles(rectangles)


if __name__ == '__main__':
    # ostd method

    # source_path, target_path = '/Volumes/datasets/chicken_count', '/Volumes/datasets/chicken_count_process'
    #
    #
    # files = []
    # for root, _, filenames in os.walk(source_path):
    #     for filename in filenames:
    #         if filename.endswith('jpg'):
    #             files.append((root, filename))
    #
    # bar = tqdm(total=len(files), desc='total')
    # update = lambda *args: bar.update()
    #
    # pool = ThreadPool(10)
    #
    #
    # for root, filename in sorted(files):
    #     kwds = {'root':root, 'filename':filename}
    #     # video2frame(**kwds)
    #     pool.apply_async(operation, kwds=kwds, callback=update)
    #
    # pool.close()
    # pool.join()

    # pipe_line, pipe_blur = find_line(
    #     '/Volumes/SoberSSD/SSD_Download/chicken/chicken_counting_clip/0_8_IPC1_20220905000000/0_8_IPC1_20220905000000_00000001.jpg')
    #
    # find_contours(pipe_line)

    test()