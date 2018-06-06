"""
here I use darknet to do detection for a video and store the result in another video, so
1. use opencv to read the video and obtain video information, such as video length, fps, image height and width
2. create another video writer to store the video after detection

In the code below, there is a code block which are commented, if it is uncommented, you can stop the video and resume
again or process frame by frame, I write this based on the viewer in the code of deep sort.

There is another thing I need to mention, the python API of darknet provided in the original repository can only take
image file path as one argument, but in my situation, what I get from opencv is an numpy array, so I can't use the python
API directly. Luckily, based on the discussion in https://github.com/pjreddie/darknet/issues/289, this issue is solved.
so here I can use the detect_np fucntion to do detection using darknet. The files that needs modification are
1. ./python/darknet.py
2. ./src/image.c and ./src/image.h
3 ./Makefile
"""
import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import time
import cv2
import glob


def detect(cfg_file, weight_file, video_file):

    dn.set_gpu(0)
    net = dn.load_net(cfg_file, weight_file, 0)
    meta = dn.load_meta("cfg/coco.data")

    print("video: %s" % video_file)
    cap = cv2.VideoCapture(video_file)
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    terminate, is_paused = False, False

    idx = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter("gray_processed.mp4", fourcc, fps, (width, height))


    while not terminate and cap.isOpened():
        idx += 1
        print("processing: %.4f%%" % (idx * 100.0 / video_len))
        #print("debug")

        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                break
            boxes = dn.detect_np(net, meta, frame)

        new_frame = my_plot(frame.copy(), boxes)
        video_writer.write(new_frame)

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
            boxes = dn.detect_np(net, meta, frame)
            is_paused = True
        """

    cap.release()
    video_writer.release()
    #cv2.destroyAllWindows()

def my_plot(img, boxes):

    color = (0, 0, 255)
    for i in range(len(boxes)):
        box = boxes[i]

        cls_name = box[0]
        if cls_name != "person":
            continue

        x, y, w, h = box[2]
        pt1 = int(x - w / 2), int(y - h / 2)
        pt2 = int(x + w / 2), int(y + h / 2)
        if pt1[0] <= 520 and pt1[1] <= 70:
            continue
        if h / w >= 6.0:
            continue

        cv2.rectangle(img, pt1, pt2, color, 2)

    return img

if __name__ == "__main__":

    if len(sys.argv) == 4:
        cfg_file = sys.argv[1]
        weight_file = sys.argv[2]
        video_file = sys.argv[3]
        detect(cfg_file, weight_file, video_file)
    else:
        print('Usage: ')
        print('  python video_demo.py cfg_file weight_file video_file')
