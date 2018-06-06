import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import cv2
import os
import glob

def detect(cfgfile, weightfile, video_file):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    use_cuda = 1
    if use_cuda:
        m.cuda()
    print("video: %s" % video_file)
    cap = cv2.VideoCapture(video_file)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    terminate, is_paused = False, False
    """
    idx = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("gray_processed.mp4", fourcc, fps, (width, height))
    """

    while not terminate and cap.isOpened():
        #idx += 1
        #print("processing: %.4f%%" % (idx * 100.0 / video_len))
        #print("debug")

        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                #print("ret is False")
                break
            sized = cv2.resize(frame, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)


        new_frame = my_plot(frame.copy(), boxes)
        #video_writer.write(new_frame)

        #"""
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
            sized = cv2.resize(frame, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
            boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
            is_paused = True
        #"""

    cap.release()
    #video_writer.release()
    cv2.destroyAllWindows()

def my_plot(img, boxes):

    height, width = img.shape[:2]

    for i in range(len(boxes)):
        box = boxes[i]

        cls_id = box[6]
        if cls_id != 0:
            continue

        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height
        # tailored for one video
        if x1 <= 520 and y1 <= 70:
            continue
        if box[3] / box[2] >= 6.0:
            continue
        #print("w:%d, h:%d, h/w:%.2f" % (int(box[2]*width), int(box[3]*height), box[3] / box[2]))
        pt1 = int(x1), int(y1)
        pt2 = int(x2), int(y2)

        color = (0, 0, 255)
        cv2.rectangle(img, pt1, pt2, color, 2)

    return img

if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        video_file = sys.argv[3]
        detect(cfgfile, weightfile, video_file)
    else:
        print('Usage: ')
        print('  python video_demo.py cfg_file weight_file video_file')
        # detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
