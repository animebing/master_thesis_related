"""Perform detection on all images of a video and store the result in one video
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.io import cache_url
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--img_dir',
        help='image directory for video images',
        required=True,
        type=str
    )
    parser.add_argument(
        '--output_dir',
        help="directory to store the processed video",
        default='output_video',
        type=str
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(args):

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    # get the weight path
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    #dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    img_list = os.listdir(args.img_dir)
    img_list = sorted(img_list)
    print("image dir: %s" % args.img_dir)
    video_len = len(img_list)
    fps = 25
    width = 640
    height = 512

    #video_file = args.video_file
    #print("video: %s" % video_file)
    #cap = cv2.VideoCapture(video_file)
    #video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #fps = int(cap.get(cv2.CAP_PROP_FPS))
    #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    terminate, is_paused = False, False


    idx = 0
    # I have to compute fourcc in advance, because there will be error
    # Expected single character string for argument 'c1', I have't figure out how to solve it
    #fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # XVID: 1145656920
    # mp4v: 1983148141
    fourcc = 1983148141
    #video_basename = os.path.basename(video_file)
    #video_name, video_ext = os.path.splitext(video_basename)
    #output_name = video_name + '_processed' + video_ext
    output_name = 'downtown_night_new.mp4'
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, output_name)
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while not terminate:
        print("processing: %.4f%%" % ((idx+1) * 100.0 / video_len))
        #print("debug")

        if not is_paused:
            img_path = os.path.join(args.img_dir, img_list[idx])
            frame = cv2.imread(img_path)
            idx += 1
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, frame, None, None
                )

        new_frame = vis_utils.vis_one_image_opencv(
                    frame,
                    cls_boxes,
                    thresh=0.7,
                    show_box=True
        )
        video_writer.write(new_frame)
        if idx == video_len:
            terminate = True
        """
        cv2.imshow('image', new_frame)
        key = cv2.waitKey(1)

        if key & 255 == 27:  # ESC
            print("terminating")
            terminate = True
        elif key & 255 == 32:  # ' '
            print("toggeling pause: " + str(not is_paused))
            is_paused = not is_paused
        elif key & 255 == 115:  # 's'
            print("stepping")
            ret, frame = cap.read()
            if not ret:
                break
            with c2_utils.NamedCudaScope(0):
                cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                    model, frame, None, None
                )
            is_paused = True
        """

    video_writer.release()
    #cv2.destroyAllWindows()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)

    """
    CUDA_VISIBLE_DEVICES=0 python2 tools/imgs2video_demo.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml \
    --wts /home/bingbing/git/detectron/output/kaist/train/train/kaist_train/generalized_rcnn/model_final.pkl \
    --img_dir /home/bingbing/datasets/kaist/set05/V000/visible \
    --output_dir output_video/
    """