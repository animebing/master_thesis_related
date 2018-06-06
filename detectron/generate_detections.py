"""Do detection on all images of one directory, then store the detection result on one txt file.
because I am not famaliar with Detectron and in order to use the API it provides, I add a new function
named write_txt in ./lib/utils/vis.py to store the detection result in one txt file

Because I just use Detectron to detect pedestrian, so I make some modifications in ./lib/utils/vis.py
to exclude other categories.
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
        '--input_dir',
        help='the path to the input directory',
        required=True,
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


    det_dir = os.path.join(args.input_dir, 'det')
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)

    txt_file = os.path.join(det_dir, "det.txt")
    fid = open(txt_file, 'w')

    img_dir = os.path.join(args.input_dir, "img1")
    img_list = os.listdir(img_dir)
    img_list = sorted(img_list)

    for i in range(len(img_list)):
        print("processing: %d/%d" % (i+1, len(img_list)))
        img_name = img_list[i][:-4]
        img_idx = int(img_name)
        img_path = os.path.join(img_dir, img_list[i])
        frame = cv2.imread(img_path)

        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, frame, None, None)

        vis_utils.write_txt(fid, img_idx, cls_boxes)

    fid.close()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)

    """
    CUDA_VISIBLE_DEVICES=0 python2 tools/generate_detections.py \
    --cfg configs/12_2017_baselines/e2e_mask_rcnn_R-50-FPN_2x.yaml \
    --wts ./model/e2e-R-50-FPN-Mask/model_final.pkl \
    --input_dir ./class_frames_maskrcnn
    """