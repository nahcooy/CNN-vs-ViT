# coding=utf-8
# @Project  ：SAFECount
# @FileName ：video_clip.py
# @Author   ：SoberReflection
# @Revision : sober
# @Date     ：2023/1/31 16:06
import multiprocessing
import os
import os.path as osp
import time
from threading import Thread
from multiprocessing import get_context
import cv2
from tqdm import tqdm, trange


def get_frame(cap, currentframe, target_path, filename):
    _, frame = cap.retrieve()
    if frame is None:
        return
    cv2.imwrite(os.path.join(target_path, '%s_%08d.jpg'%(filename, currentframe)), frame)



def video2frame(path, target_path, interval = 10, bar = None):
    """
    @param path:
    @param target_path:
    @param interval: how much second one image
    @return:
    """
    basename = osp.splitext(osp.basename(path))[0]
    target_path = osp.join(target_path, basename)
    os.makedirs(target_path, exist_ok=True)

    cap = cv2.VideoCapture(path)
    filename = osp.splitext(osp.basename(path))[0]

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")
        return

    # Read until video is completed
    currentframe = 0
    frameRate = cap.get(5)  # frame rate
    frameLength = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # bar = tqdm(total = round(frameLength / frameRate / interval), desc='%s'%filename)

    # print('FrameRate and frameLength, ', frameRate, frameLength)

    # for index in trange(int(frameLength), desc='%s'%filename):
    for index in range(int(frameLength)):
        ret = cap.grab()
        if not ret:
            break
        if index % int(frameRate * interval) == 0:
            # get_frame(cap, currentframe, target_path, filename)
            Thread(target=get_frame, args=(cap, currentframe, target_path, filename), daemon=True).start()
            currentframe += 1

    cap.release()
    # Closes all the frames

    if bar is not None:
        bar.update()


if __name__ == '__main__':
    source_p, target_p = '/Volumes/SoberSSD/SSD_Download/chicken/no_chicken', \
                         '/Volumes/SoberSSD/SSD_Download/chicken/no_chicken_clip'


    all_files = []
    for root, dirs, files in os.walk(source_p):
        for file in files:
            if file.endswith('.mp4') and not file.startswith('.'):
                all_files.append(osp.join(root, file))


    bar = tqdm(total=len(all_files), desc='total')
    update = lambda *args: bar.update()

    pool = multiprocessing.Pool(2)


    for path in sorted(all_files):
        kwds = {'path':path, 'target_path':target_p, 'interval':20}
        # video2frame(**kwds)
        pool.apply_async(video2frame, kwds=kwds, callback=update)

    pool.close()
    pool.join()


