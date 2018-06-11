# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np
import time

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from yolo3.darknet import Darknet
from yolo3.utils import do_detect

total_frames = 0
total_time = 0.0

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    # NOTE: what is feature_dim used for? -> which is appearance feature related to each detection
    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


#def create_detections(detection_mat, frame_idx, image_filename, encoder, min_height=0):
def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    #bgr_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)
    #boxes = detection_mat[mask][:, 2:6]
    #features = encoder(bgr_image, boxes.copy())

    detection_list = []
    #for row, appearance in zip(detection_mat[mask], features):
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        #bbox, confidence, feature = row[2:6], row[6], appearance
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def create_gts(gt_mat, frame_idx):
    """Create detections for given frame index from the raw groundtruth detection matrix.

    Parameters
    ----------
    gt_mat : ndarray
        Matrix of ground truth detections
    frame_idx : int
        The frame index.

    Returns
    -------
    List[detection.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = gt_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    #for row, appearance in zip(detection_mat[mask], features):
    for row in gt_mat[mask]:
        if int(row[6]) == 0 or int(row[7]) != 1:
            continue
        bbox = row[2:6]
        feature = np.ones((1,), dtype=np.float32)
        vis = float(row[-1])
        #print('create gt detections')
        assert int(row[6]) == 1, 'flag should be 1'
        #if bbox[3] < min_height:
        #    continue
        detection_list.append(Detection(bbox, vis, feature))
    return detection_list


def create_model(cfg_file, weight_file, use_cuda):
    model = Darknet(cfg_file)
    model.print_network()
    model.load_weights(weight_file)

    if use_cuda:
        model.cuda()
    return model


def create_det_from_model(model, img, conf_thresh, nms_thresh, min_detection_height, use_cuda):

    height, width = img.shape[:2]
    sized = cv2.resize(img, (model.width, model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(model, sized, conf_thresh, nms_thresh, use_cuda)

    dummy_feat = np.ones((1,), dtype=np.float32)
    detection_list = []

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = box[6]
        if cls_id != 0:
            continue

        x = int(round((box[0] - box[2] / 2.0) * width))
        y = int(round((box[1] - box[3] / 2.0) * height))
        w = int(box[2] * width)
        h = int(box[3] * height)
        if h < min_detection_height:
            continue
        cls_conf = box[5]

        detection_list.append(Detection((x, y, w, h), cls_conf, dummy_feat))

    return detection_list


def run(sequence_dir, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """

    global total_frames, total_time

    cfg_file = "yolo3/cfg/yolov3.cfg"
    #weight_file = "yolo3/yolov3.weights"
    weight_file = 'yolo3/backup/MOT17Det/yolov3-mot17det_10000.weights'
    use_cuda = 1


    det_model = create_model(cfg_file, weight_file, use_cuda)

    seq_info = gather_sequence_info(sequence_dir, None)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    #metric = nn_matching.NearestNeighborDistanceMetric(
    #    "euclidean", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    img = cv2.imread(seq_info["image_filenames"][1])

    sized = cv2.resize(img, (det_model.width, det_model.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(det_model, sized, 0.5, 0.4, use_cuda)


    print("processing: %s" % os.path.basename(sequence_dir))

    def frame_callback(vis, frame_idx):
        #print("Processing frame %05d" % frame_idx)
        global total_frames, total_time
        img = cv2.imread(seq_info["image_filenames"][frame_idx])

        time_0 = time.time()
        # Load image and generate detections.
        detections = create_det_from_model(det_model, img, 0.5, 0.4, min_detection_height, use_cuda)

        #detections = create_detections(
        #    seq_info["detections"], frame_idx, min_detection_height)
        #if seq_info['groundtruth'] is not None:
        #    gts = create_gts(seq_info['groundtruth'], frame_idx)

        #detections = create_detections(
        #    seq_info["detections"], frame_idx, seq_info["image_filenames"][frame_idx], encoder, min_detection_height)


        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        #tracker.predict()
        #tracker.update(detections)
        time_1 = time.time()
        total_time += time_1 - time_0
        total_frames += 1
        # Update visualization.
        if display:
            #image = cv2.imread(
            #    seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            #vis.set_image(image.copy())
            vis.set_image(img.copy())
            vis.draw_detections(detections)
            #vis.draw_detections(gts)
            #vis.draw_trackers(tracker.tracks)

        # Store results.
        # NOTE: store from n_init frame(1-based index)


        for track in tracker.tracks:
            # NOTE: the condition is different from that in drawing tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            # NOTE: store estimated state instead of observation
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])


    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=1)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    print(
        "average time for one frame: %.5f, FPS: %.5f" % (total_time / total_frames, total_frames / total_time))
    # Store results.
    #f = open(output_file, 'w')
    #for row in results:
    #    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
    #        row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

    """
    with open(output_file, 'w') as f:
        for row in results:
            print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                row[0], row[1], row[2], row[3], row[4], row[5]),file=f)
    """

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    #parser.add_argument(
    #    "--model",
    #    default="resources/networks/mars-small128.pb",
    #    help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    #parser.add_argument(
    #    "--detection_file", help="Path to custom detections.", default=None,
    #    required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    #parser.add_argument(
    #    "--display", help="Show intermediate tracking results",
    #    default=False, type=bool)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    run(
        args.sequence_dir, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)

    """
    python deep_sort_yolo3.py \
    --sequence_dir=./MOT17/train/MOT17-04-YOLO \
    --min_confidence=0.1 \
    --nn_budget=100 \
    """
