from __future__ import print_function
import os
import cv2
import argparse
import glob
import numpy as np
import time

from utils import *
from darknet import Darknet

def generate_det(seq_dir, npy_dir, cfg_file, weight_file):

    # load yolo model
    m = Darknet(cfg_file)
    m.print_network()
    m.load_weights(weight_file)
    print('Loading weights from %s... Done!' % (weight_file))

    use_cuda = 1
    if use_cuda:
        m.cuda()

    if not os.path.exists(npy_dir):
        os.makedirs(npy_dir)

    seq_list = glob.glob(os.path.join(seq_dir, "*-YOLO"))
    seq_list = sorted(seq_list)

    for seq in seq_list:
        seq_name = os.path.basename(seq)
        print("processing: %s" % seq_name)
        det_dir = os.path.join(seq, "det")
        if not os.path.exists(det_dir):
            os.makedirs(det_dir)
        txt_file = os.path.join(det_dir, "det.txt")
        fid = open(txt_file, 'w')

        npy_file = os.path.join(npy_dir, seq_name + ".npy")

        img_dir = os.path.join(seq, "img1")
        img_list = os.listdir(img_dir)
        img_list = sorted(img_list)

        img = cv2.imread(os.path.join(img_dir, img_list[0]))
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

        total_time = 0.0

        npy_list = []
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
                npy_list.append([img_idx, -1.0, x, y, w, h, cls_conf, -1.0, -1.0, -1.0, 1.0])

        fid.close()
        np.save(npy_file, np.asarray(npy_list, dtype=np.float32),allow_pickle=False)
        #print("average time for one frame: %.5f, FPS: %.5f" % (
        #    total_time / len(img_list), len(img_list) / total_time))

def parse_args():

    parser = argparse.ArgumentParser(description="generate detection result of MOT17 train using yolo3")

    parser.add_argument("--seq_dir", help="Path to MOTChallenge sequence directory",
                        required=True)

    parser.add_argument("--cfg_file", help="config file of yolo",
                        default="cfg/yolov3.cfg")
    parser.add_argument("--weight_file", help="weight file of yolo",
                       default="yolov3.weights")
    parser.add_argument("--npy_dir", help="directory to store numpy file for tracking use",
                        default="npy_files")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_det(args.seq_dir, args.npy_dir, args.cfg_file, args.weight_file)

    """
    python generate_detections_npy.py --seq_dir ./MOT17/train --npy_dir ./npy_files --cfg_file ./cfg/yolov3.cfg \
                            --weight_file ./backup/MOT17Det/yolov3-mot17det_10000.weights
    """