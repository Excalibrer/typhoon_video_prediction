from moviepy.editor import *
from settings import *
import cv2
import os
import matplotlib.pyplot as plt
from skimage import io
import glob
import imageio

video_path = 'typhoon_video'
store_path = 'typhoon_data'

def Video_Clip(video_path, video_name, new_video_name, start_from, end_at):
    '''
    :param video_path: 存放视频的路径，包括源视频和新视频
    :param video_name: 源视频名称
    :param new_video_name: 新视频命名的名称
    :param start_from: 开始截取的时间戳
    :param end_at: 结束截取的时间戳
    :return:
    '''
    clip = VideoFileClip(os.path.join(video_path, video_name)).subclip(start_from, end_at)
    clip.write_videofile(os.path.join(video_path, new_video_name))

Video_Clip(video_path=video_path, video_name='typhoon_01.flv',
           new_video_name='tc_01.mp4', start_from=12, end_at=22)
Video_Clip(video_path=video_path, video_name='typhoon_02.flv',
           new_video_name='tc_02.mp4', start_from=17, end_at=30)
Video_Clip(video_path=video_path, video_name='typhoon_03.flv',
           new_video_name='tc_03.mp4', start_from=23, end_at=33)
Video_Clip(video_path=video_path, video_name='chaos.flv',
           new_video_name='chaos.mp4', start_from=80, end_at=87)

def video2image(video_path, video_name, store_path, time):
    '''
    :param video_path: 需要处理的视频路径
    :param video_name: 需要处理的视频名字
    :param store_path: 存放帧图的路径
    :param time: 每time个帧图保存一张帧图
    :return:
    '''
    save_path = os.path.join(store_path, video_name.split('.')[0])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    vc = cv2.VideoCapture(os.path.join(video_path, video_name))
    c = 1
    p = 1
    rval = vc.isOpened()
    while rval:
        rval, fname = vc.read()
        if(c % time == 0):
            cv2.imwrite(os.path.join(save_path, 'tc_%03d.jpg' % p), fname)
            p += 1
        c += 1
    vc.release()

video2image(video_path=video_path, video_name='tc_01.mp4', store_path=store_path, time=10)
video2image(video_path=video_path, video_name='tc_02.mp4', store_path=store_path, time=10)
video2image(video_path=video_path, video_name='tc_03.mp4', store_path=store_path, time=10)
video2image(video_path=video_path, video_name='test.mp4', store_path=store_path, time=5)
video2image(video_path=video_path, video_name='chaos.mp4', store_path=store_path, time=10)
video2image(video_path=video_path, video_name='test_1.mp4', store_path=store_path, time=10)

def frame_cut(path):
    '''
    :param path: 图片的母文件夹路径
    :return:
    '''
    img_paths = glob.glob(os.path.join(path, '*.jpg'))
    for img_path in img_paths:
        img = io.imread(img_path)
        d = img_path.split('.')[0].split('_')[-1]
        img = img[100:700, :1300]
        io.imsave(os.path.join(path, 'tc_%03d.png' % int(d)), img)
        os.remove(img_path)

frame_cut(path='typhoon_data/tc_01')
frame_cut(path='typhoon_data/tc_02')
frame_cut(path='typhoon_data/tc_03')

def create_gif(gif_name, path, duration = 0.3):
    '''
    生成gif文件，原始图片仅支持png格式
    gif_name ： 字符串，所生成的 gif 文件名，带 .gif 后缀
    path :      需要合成为 gif 的图片所在路径
    duration :  gif 图像时间间隔
    '''

    frames = []
    pngFiles = sorted(os.listdir(path))
    image_list = [os.path.join(path, f) for f in pngFiles if f != '.DS_Store']
    for image_name in image_list:
        frames.append(io.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration = duration)
    return

create_gif(gif_name='tc_01.gif', path='typhoon_data/tc_01', duration=0.5)
create_gif(gif_name='tc_02.gif', path='typhoon_data/tc_02', duration=0.5)
create_gif(gif_name='tc_03.gif', path='typhoon_data/tc_03', duration=0.5)
create_gif(gif_name='test.gif', path='typhoon_data/test', duration=0.3)
create_gif(gif_name='chaos.gif', path='typhoon_data/chaos', duration=0.3)
create_gif(gif_name='test_1.gif', path='typhoon_data/test_1', duration=0.3)

desired_im_sz = (128, 160)

test_recordings = ['typhoon_4', 'typhoon_6', 'typhoon_10']
val_recordings = ['typhoon_5']

import numpy as np
from scipy.misc import imread, imresize
import hickle as hkl
import re

def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    not_train = splits['val'] + splits['test']
    for c in os.listdir(DATA_DIR):
        if c not in not_train and re.match(r'typhoon*', c):
            splits['train'].append(c)

    # 用im_list每个元素都是一个子列表，子列表存放一个文件夹内图片的路径，
    # 用source_list每个元素都是一个子列表，子列表存放所有帧图的类别，
    # im_list和source_list的长度为train集的大小*每个文件夹图片的数量
    for split in splits:
        im_list = []
        source_list = []  # corresponds to recording that image came from
        for folder in splits[split]:
            im_dir = os.path.join(DATA_DIR, folder+'/')
            files = []
            for _, _, file in os.walk(im_dir):
                if '.DS_Store' in file:
                    file.remove('.DS_Store')
                files.append(file)
            im_list += [im_dir + f for f in sorted(files[0])]
            source_list += [folder] * len(files[0])

        print('Creating ' + split + ' data: ' + str(len(im_list)) + ' images')
        X = np.zeros((len(im_list),) + desired_im_sz + (3,), np.uint8)
        for i, im_file in enumerate(im_list):
            im = imread(im_file)
            X[i] = process_im(im, desired_im_sz)
        print('Create Data Finished')
        hkl.dump(X, os.path.join(DATA_DIR, 'Typhoons_X_' + split + '.hkl'))
        hkl.dump(source_list, os.path.join(DATA_DIR, 'Typhoons_sources_' + split + '.hkl'))
        print('Store Finished')

def process_im(im, desired_sz):
    target_ds = float(desired_sz[0])/im.shape[0]
    im = imresize(im, (desired_sz[0], int(np.round(target_ds * im.shape[1]))))
    d = int((im.shape[1] - desired_sz[1]) / 2)
    im = im[:, d:d+desired_sz[1]]
    return im

process_data()