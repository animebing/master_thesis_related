from __future__ import print_function
import os
import cv2
import argparse
import glob
import numpy as np
import time

from utils import *
from darknet import Darknet

def generate_det(input_dir, cfg_file, weight_file):

    # load yolo model
    m = Darknet(cfg_file)
    m.print_network()
    m.load_weights(weight_file)
    print('Loading weights from %s... Done!' % (weight_file))

    use_cuda = 1
    if use_cuda:
        m.cuda()

    det_dir = os.path.join(input_dir, 'det')
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)

    txt_file = os.path.join(det_dir, "det.txt")
    fid = open(txt_file, 'w')


    img_dir = os.path.join(input_dir, "img1")
    img_list = os.listdir(img_dir)
    img_list = sorted(img_list)

    img = cv2.imread(os.path.join(img_dir, img_list[0]))
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

    #total_time = 0.0

    for i in range(len(img_list)):
        print("processing: %d/%d" % (i+1, len(img_list)))
        img_name = img_list[i][:-4]
        img_idx = int(img_name)
        img_path = os.path.join(img_dir, img_list[i])

        img = cv2.imread(img_path)
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        #time_0 = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        #time_1 = time.time()
        #total_time += time_1 - time_0


        height, width = img.shape[:2]
        for j in range(len(boxes)):
            box = boxes[j]
            cls_id = box[6]
            if cls_id != 0:
                continue

            x = (box[0] - box[2] / 2.0) * width
            y = (box[1] - box[3] / 2.0) * height
            w = box[2] * width
            h = box[3] * height
            cls_conf = box[5]

            txt_tmp = "%d,-1,%.1f,%.1f,%.1f,%.1f,%.3f\n" % (img_idx, x, y, w, h, cls_conf)
            fid.write(txt_tmp)

    fid.close()

def parse_args():

    parser = argparse.ArgumentParser(description="generate detection result of MOT17 train using yolo3")

    parser.add_argument("--input_dir", help="Path to the input directory",
                        required=True)

    parser.add_argument("--cfg_file", help="config file of yolo",
                        default="cfg/yolov3.cfg")
    parser.add_argument("--weight_file", help="weight file of yolo",
                       default="yolov3.weights")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_det(args.input_dir, args.cfg_file, args.weight_file)

    """
    python generate_detections_txt.py --input_dir ./class_frames  --cfg_file ./cfg/yolov3.cfg \
                            --weight_file ./yolov3.weights
    """