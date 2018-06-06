import sys
import time
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet
import cv2
import os
import glob

def detect(cfgfile, weightfile, img_dir):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 1
    if use_cuda:
        m.cuda()

    img_list = glob.glob(os.path.join(img_dir, '*.jpg'))
    img_list = sorted(img_list)

    # warming up
    img = cv2.imread(img_list[0])
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)

    class_names = load_class_names(namesfile)

    for i in range(len(img_list)):

        img = cv2.imread(img_list[i])
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        start = time.time()
        boxes = do_detect(m, sized, 0.5, 0.4, use_cuda)
        finish = time.time()
        #my_plot_boxes(img, boxes, class_names)
        print("processing time: %.5f" % (finish - start))

    cv2.destroyAllWindows()
    # print("average time for 30 test: %.5f" % (total_time / 30))
    # print("average fps: %.2f" % (30 / total_time))



def my_plot_boxes(img, boxes, class_names):

    height, width = img.shape[:2]

    for i in range(len(boxes)):
        box = boxes[i]
        x1 = (box[0] - box[2]/2.0) * width
        y1 = (box[1] - box[3]/2.0) * height
        x2 = (box[0] + box[2]/2.0) * width
        y2 = (box[1] + box[3]/2.0) * height
        pt1 = int(x1), int(y1)
        pt2 = int(x2), int(y2)
        #cls_conf = box[5]
        cls_id = box[6]
        cls_name = class_names[cls_id]

        color = (0, 0, 255)
        cv2.rectangle(img, pt1, pt2, color, 2)
        text_size = cv2.getTextSize(cls_name, cv2.FONT_HERSHEY_PLAIN, 2, 2)
        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
        cv2.rectangle(img, pt1, pt2, color, -1)
        cv2.putText(img, cls_name, center, cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # cv2.imwrite('detection.png', img)
    cv2.imshow('image', img)
    cv2.waitKey(1)

if __name__ == '__main__':
    if len(sys.argv) == 4:
        cfgfile = sys.argv[1]
        weightfile = sys.argv[2]
        img_dir = sys.argv[3]
        detect(cfgfile, weightfile, img_dir)
        # detect_cv2(cfgfile, weightfile, imgfile)
        # detect_skimage(cfgfile, weightfile, imgfile)
    else:
        print('Usage: ')
        print('  python detect.py cfgfile weightfile imgfile')
        # detect('cfg/tiny-yolo-voc.cfg', 'tiny-yolo-voc.weights', 'data/person.jpg', version=1)
