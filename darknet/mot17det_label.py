"""
I am trying to use MOT17Det dataset to train yolov3, so first I need to make the annotation files.
if $DATASET is the dataset directory, then all images in $DATASET/images, all labels in $DATASET/labels.
In darknet, for each image file, there is one lable txt file which has the same name as the image file without
file extension. 

In addition to these two directories, we still need a txt file which contains a image file path in each line 
for each image and the file path is the absolute path.

for training MOT17DET, I need to create a labels directory for each video, then create a txt file which contains
the paths of all images of all videos. Because in MOT17DET all images of a video is in a directory named "img1", in 
order to make it compatible with darknet form, there are two solutions:
    1. make a soft link named images for each img1 directory
    2. make img1 one of options in darknet, in order to make it, you need to make some modifications to src/data.c,
    in the "fill_truth_detection" function, add "find_replace(labelpath, "img1", "labels", labelpath);" after 
    "find_replace(labelpath, "raw", "labels", labelpath);", then "make" again. In data.c, you will also find there 
    are some another optional image directories you can use

in my implementation, I use solution 2. When you want to use darknet for another dataset later, this modification is 
also useful.
"""
from __future__ import print_function
import os
import numpy as np
import cv2

def convert(size, bbox):
    img_w, img_h = size
    dw = 1. / img_w
    dh = 1. / img_h

    x, y, w, h = bbox
    x_c = x + w / 2.0 - 1
    y_c = y + h / 2.0 -1

    x_c = x_c * dw
    w = w * dw
    y_c = y_c * dh
    h = h * dh
    return (x_c, y_c, w, h)

def convert_annotation(label_txt, anno_data, img_w, img_h):
    label_fid = open(label_txt, 'w')

    for row in anno_data:
        if int(row[6]) == 0 or int(row[7]) != 1 or float(row[-1]) < 0.2:
            continue
        bbox = row[2:6]
        cls_id= 0
        bb = convert((img_w, img_h), bbox)
        label_fid.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

    label_fid.close()


store_dir = '/home/bingbing/git/darknet/MOT17Det'
data_dir = os.path.join(store_dir, 'train')

seq_list = os.listdir(data_dir)
seq_list = sorted(seq_list)

train_txt = os.path.join(store_dir, 'train.txt')
train_fid = open(train_txt, 'w')
val_txt = os.path.join(store_dir, 'val.txt')
val_fid = open(val_txt, 'w')

for seq in seq_list:
    seq_dir = os.path.join(data_dir, seq)
    if not os.path.isdir(seq_dir):
        continue
    print("processing: %s" % seq)

    img_dir = os.path.join(seq_dir, 'img1')
    label_dir = os.path.join(seq_dir, 'labels')
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    gt_txt = os.path.join(seq_dir, 'gt/gt.txt')
    gt_data = np.loadtxt(gt_txt, delimiter=',')
    frame_indices = gt_data[:, 0].astype(np.int)

    img_list = os.listdir(img_dir)
    train_num = int(len(img_list) * 0.9)
    img = cv2.imread(os.path.join(img_dir, img_list[0]))
    img_h, img_w = img.shape[:2]
    print("train: %d, val: %d" % (train_num, len(img_list) - train_num))

    for i in range(train_num):
        img_name = img_list[i][:-4]
        frame_idx = int(img_name)
        img_file = os.path.join(img_dir, img_list[i])
        label_txt = os.path.join(label_dir, img_name + ".txt")

        mask = frame_indices == frame_idx

        convert_annotation(label_txt, gt_data[mask], img_w, img_h)
        train_fid.write(img_file + '\n')

    for i in range(train_num, len(img_list)):
        img_name = img_list[i][:-4]
        frame_idx = int(img_name)
        img_file = os.path.join(img_dir, img_list[i])
        label_txt = os.path.join(label_dir, "img_name" + ".txt")

        mask = frame_indices == frame_idx

        convert_annotation(label_txt, gt_data[mask], img_w, img_h)
        val_fid.write(img_file + '\n')

train_fid.close()
val_fid.close()

